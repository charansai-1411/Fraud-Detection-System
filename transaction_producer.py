import pandas as pd
import json
import time
import sys
import os
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

# Kafka configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions")

def get_producer(broker_address, retries=5, delay=5):
    """
    Attempts to establish a connection to the Kafka Broker.
    Retries multiple times in case Kafka is still starting up.
    """
    print(f"Connecting to Kafka broker at {broker_address}...")
    for i in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[broker_address],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("Successfully connected to Kafka!")
            return producer
        except NoBrokersAvailable:
            print(f"[{i+1}/{retries}] Broker not available. Retrying in {delay} seconds...")
            time.sleep(delay)
    print("Could not connect to Kafka Broker. Exiting.")
    sys.exit(1)

def find_sample_csv():
    """
    Finds the sample CSV file in the current directory or parent directory.
    """
    candidate_files = [
        "sample_transactions_with_fraud(1).csv",
        "sample_transactions (1).csv",
        "sample_transactions.csv"
    ]
    
    # Check current directory
    for f in candidate_files:
        if os.path.exists(f):
            return f
            
    # Check parent directory
    for f in candidate_files:
        parent_path = os.path.join("..", f)
        if os.path.exists(parent_path):
            return parent_path
            
    # Fallback search
    for f in os.listdir('.'):
        if f.endswith('.csv'):
            return f
            
    return None

def main():
    # 1. Find CSV File
    csv_file = find_sample_csv()
    if not csv_file:
        print("❌ Error: No candidate CSV file found in the workspace directory.")
        print("Please place a CSV file containing transaction data in this directory.")
        sys.exit(1)
        
    print(f"📂 Found transaction CSV: {csv_file}")
    
    # 2. Load Data
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} transactions.")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        sys.exit(1)
        
    # 3. Setup Kafka Producer
    producer = get_producer(KAFKA_BROKER)
    
    # 4. Stream Transactions
    print(f"🚀 Starting real-time simulation. Publishing to topic '{KAFKA_TOPIC}'...")
    print("Press Ctrl+C to stop.\n")
    
    count = 0
    try:
        while True:
            # Shuffle or loop indefinitely to simulate continuous stream
            for index, row in df.iterrows():
                # Convert Series row to clean dictionary
                data = row.to_dict()
                
                # Make sure fields are standard python types for JSON serialization
                for key, val in data.items():
                    if hasattr(val, 'item'):  # converts numpy types to python native types
                        data[key] = val.item()
                
                # Send to Kafka
                producer.send(KAFKA_TOPIC, data)
                count += 1
                
                amount = data.get("Amount", 0.0)
                actual_class = int(data.get("Class", -1))
                class_str = " (FRAUD)" if actual_class == 1 else ""
                
                print(f"📤 Sent transaction #{count:04d}: Amount = ${amount:.2f}{class_str}")
                
                # Pause to simulate real-world transaction rate (approx 5/sec)
                time.sleep(0.2)
                
            print("🔄 Finished CSV dataset. Looping again for continuous simulation...")
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")
    finally:
        producer.flush()
        producer.close()
        print("Producer closed.")

if __name__ == "__main__":
    main()
