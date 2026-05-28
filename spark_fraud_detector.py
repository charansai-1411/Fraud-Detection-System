import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, struct, from_json, lit, to_json
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType
from pyspark import SparkFiles

# Kafka settings
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "transactions")
OUTPUT_TOPIC = os.getenv("OUTPUT_TOPIC", "fraud_alerts")

# Initialize Spark Session with Kafka Package integration
print("Initializing PySpark Session with Kafka Connector...")
spark = SparkSession.builder \
    .appName("FraudDetectionStreaming") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .config("spark.sql.shuffle.partitions", "2") \
    .getOrCreate()

# Add XGBoost model, scaler, and config files to SparkContext
# This distributes them to all workers for local execution.
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "fraud_model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")
config_path = os.path.join(current_dir, "model_config.json")

print(f"Distributing resources to Spark workers:\n - Model: {model_path}\n - Scaler: {scaler_path}\n - Config: {config_path}")
spark.sparkContext.addFile(model_path)
spark.sparkContext.addFile(scaler_path)
spark.sparkContext.addFile(config_path)

# Define UDF execution helper
_model = None
_scaler = None
_threshold = None

def get_resources():
    global _model, _scaler, _threshold
    if _model is None or _scaler is None:
        import pickle
        import json
        
        # Load resources using SparkFiles to get distributed absolute paths
        m_file = SparkFiles.get("fraud_model.pkl")
        s_file = SparkFiles.get("scaler.pkl")
        c_file = SparkFiles.get("model_config.json")
        
        with open(m_file, 'rb') as f:
            _model = pickle.load(f)
        with open(s_file, 'rb') as f:
            _scaler = pickle.load(f)
            
        try:
            with open(c_file, 'r') as f:
                cfg = json.load(f)
                _threshold = float(cfg.get("best_threshold", 0.01))
        except Exception:
            _threshold = 0.01
            
    return _model, _scaler, _threshold

# Define UDF return schema
prediction_schema = StructType([
    StructField("probability", DoubleType(), False),
    StructField("prediction", IntegerType(), False),
    StructField("risk_level", StringType(), False)
])

@udf(returnType=prediction_schema)
def predict_fraud(*cols):
    """
    UDF running XGBoost and Preprocessing scaling inside PySpark execution environment.
    Columns order: V1 to V28, Amount, Time
    """
    model, scaler, threshold = get_resources()
    
    v_cols = list(cols[:28])
    amount = float(cols[28])
    time_val = float(cols[29])
    
    import pandas as pd
    import numpy as np
    
    # 1. Scale Amount and Time using the pre-fitted training standard scalers
    amount_scaled = scaler['amount_scaler'].transform([[amount]])[0][0]
    time_scaled = scaler['time_scaler'].transform([[time_val]])[0][0]
    
    # 2. Build model features in exact training order
    features = v_cols + [amount_scaled, time_scaled]
    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
    
    # 3. Create Pandas DataFrame for inference to preserve feature names
    features_df = pd.DataFrame([features], columns=feature_names)
    
    # 4. Predict
    prob = float(model.predict_proba(features_df)[0][1])
    pred = 1 if prob >= threshold else 0
    risk = "HIGH" if prob >= 0.8 else "MEDIUM" if prob >= 0.5 else "LOW"
    
    return (prob, pred, risk)

def main():
    # Define JSON schema for incoming transactions
    json_schema = StructType([
        StructField("Time", DoubleType(), True),
        StructField("Amount", DoubleType(), True),
        *[StructField(f"V{i}", DoubleType(), True) for i in range(1, 29)],
        StructField("Class", IntegerType(), True)
    ])

    print(f"Subscribing to Kafka topic '{INPUT_TOPIC}'...")
    
    # Read raw JSON messages from Kafka
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("subscribe", INPUT_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()

    # Cast key/value to String and parse value as JSON
    parsed_df = kafka_df \
        .selectExpr("CAST(value AS STRING) as json_payload") \
        .select(from_json(col("json_payload"), json_schema).alias("data")) \
        .select("data.*")

    # Select columns in UDF-expected order
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    
    # Apply standard columns checking (handles nullable/missing values)
    clean_df = parsed_df.na.fill(0.0, subset=feature_cols)

    # Run XGBoost ML Prediction in Spark Worker environment
    enriched_df = clean_df.withColumn(
        "pred", 
        predict_fraud(*[col(c) for c in feature_cols])
    )

    # Output selection
    output_df = enriched_df.select(
        col("Time"),
        col("Amount"),
        col("Class") if "Class" in parsed_df.columns else lit(-1).alias("Class"),
        col("pred.probability").alias("fraud_probability"),
        col("pred.prediction").alias("prediction"),
        col("pred.risk_level").alias("risk_level")
    )

    # Prepare message values as JSON for Kafka output
    kafka_output_df = output_df.select(
        to_json(struct("*")).alias("value")
    )

    # Clean local checkpoint folder to avoid conflict permissions on Windows
    checkpoint_dir = os.path.join(current_dir, ".spark_checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print(f"Starting Structured Streaming. Writing predictions to Kafka topic '{OUTPUT_TOPIC}'...")
    
    # Start query writing back to the Kafka topic 'fraud_alerts'
    query = kafka_output_df.writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BROKER) \
        .option("topic", OUTPUT_TOPIC) \
        .option("checkpointLocation", checkpoint_dir) \
        .start()

    # Run stream and block until termination
    query.awaitTermination()

if __name__ == "__main__":
    main()
