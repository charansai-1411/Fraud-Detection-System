import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import queue
import threading
import os
import matplotlib.pyplot as plt
# import seaborn as sns  # (Not imported or needed — kept comments for legacy)

# Set page layout to wide and add premium custom page styling
st.set_page_config(
    page_title="Real-Time Fraud Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via Markdown (glowing buttons, custom dark card elements, fonts)
st.markdown("""
    <style>
    /* Custom font import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    .status-live {
        padding: 6px 12px;
        border-radius: 20px;
        background-color: rgba(220, 53, 69, 0.2);
        color: #dc3545;
        font-weight: 600;
        border: 1px solid #dc3545;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.4); }
        70% { opacity: 1; box-shadow: 0 0 0 10px rgba(220, 53, 69, 0); }
        100% { opacity: 0.6; box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
    }
    
    .title-gradient {
        background: linear-gradient(90deg, #ff4b4b, #ff7e40, #ffc837);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }
    </style>
""", unsafe_allow_html=True)

# Define thread-safe queue for sharing Kafka events
if 'event_queue' not in st.session_state:
    st.session_state.event_queue = queue.Queue()
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
if 'max_buffer' not in st.session_state:
    st.session_state.max_buffer = 150
if 'kafka_connected' not in st.session_state:
    st.session_state.kafka_connected = False

# Read best threshold from model config
@st.cache_resource
def load_best_threshold():
    try:
        with open('model_config.json', 'r') as f:
            cfg = json.load(f)
            return float(cfg.get("best_threshold", 0.01))
    except Exception:
        return 0.01

BEST_THRESHOLD = load_best_threshold()

# ── Load Model & Scaler for Self-Contained Cloud Demo ───────────────
@st.cache_resource
def load_app_resources():
    import pickle
    try:
        with open('fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        print(f"Error loading resources: {e}")
        return None, None

@st.cache_data
def load_simulation_data():
    candidate_files = [
        "sample_transactions (1).csv",
        "sample_transactions_with_fraud(1).csv",
        "sample_transactions.csv"
    ]
    for f in candidate_files:
        if os.path.exists(f):
            return pd.read_csv(f)
    return None

# ── Sidebar Configurations ─────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/nolan/128/security-shield.png", width=70)
st.sidebar.markdown("<h2 style='margin-top:0;'>🛡️ FRAUD GUARD</h2>", unsafe_allow_html=True)
st.sidebar.divider()

# Connection Mode Selection (Crucial for Cloud Deployment)
st.sidebar.subheader("🔌 Connection Mode")
system_mode = st.sidebar.radio(
    "Select Dashboard Mode",
    ["🚀 Cloud Demo (Auto-Run)", "🔌 Enterprise Kafka Stream"],
    help="Demo mode runs fully in the cloud using standard ML. Enterprise mode connects to your local or cloud Kafka+Spark stream."
)

# ── Kafka Background consumer settings ─────────────────────────────
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
ALERTS_TOPIC = os.getenv("ALERTS_TOPIC", "fraud_alerts")

def kafka_consumer_thread(broker_addr, topic_name, q):
    print("Background consumer thread started.")
    from kafka import KafkaConsumer
    consumer = None
    try:
        consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=[broker_addr],
            auto_offset_reset='latest',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        st.session_state.kafka_connected = True
        for msg in consumer:
            q.put(msg.value)
    except Exception as e:
        print(f"Kafka consumer connection error: {e}")
        st.session_state.kafka_connected = False
    finally:
        if consumer:
            consumer.close()

# Start consumer thread ONLY in Kafka mode
if system_mode == "🔌 Enterprise Kafka Stream":
    if 'consumer_started' not in st.session_state:
        st.session_state.consumer_started = True
        t = threading.Thread(
            target=kafka_consumer_thread,
            args=(KAFKA_BROKER, ALERTS_TOPIC, st.session_state.event_queue),
            daemon=True
        )
        t.start()

# Status indicator
if system_mode == "🚀 Cloud Demo (Auto-Run)":
    st.sidebar.markdown("""
        <div class="status-live" style="background-color: rgba(40, 167, 69, 0.2); color: #28a745; border: 1px solid #28a745;">
            <span style="height: 8px; width: 8px; background-color: #28a745; border-radius: 50%; display: inline-block;"></span>
            CLOUD AUTO-DEMO ACTIVE
        </div>
    """, unsafe_allow_html=True)
else:
    if st.session_state.kafka_connected:
        st.sidebar.markdown("""
            <div class="status-live">
                <span style="height: 8px; width: 8px; background-color: #dc3545; border-radius: 50%; display: inline-block;"></span>
                LIVE FEED ACTIVE
            </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.warning("🛑 Kafka offline / Connecting...")

st.sidebar.subheader("⚙️ Stream Controls")
buffer_size = st.sidebar.slider("Live table buffer size", 10, 500, 100, 10)
st.session_state.max_buffer = buffer_size

st.sidebar.divider()
st.sidebar.markdown("""
    ### 📊 System Specs
    * **Engine**: Apache Spark Streaming
    * **Classifier**: XGBoost
    * **Scaling**: StandardScaler
    * **Broker**: Apache Kafka (`9092`)
    * **Batch Interval**: Micro-batches
""")

if st.sidebar.button("Clear Dashboard Memory", type="secondary"):
    st.session_state.transactions = []
    st.rerun()

# ── Title Header ──────────────────────────────────────────────
col_title, col_status = st.columns([4, 1], vertical_alignment="center")
with col_title:
    st.markdown('<div class="title-gradient">Real-Time Fraud Guard</div>', unsafe_allow_html=True)
    st.markdown("##### Real-Time Credit Card Fraud Detection Powered by Apache Kafka & Spark Structured Streaming")
with col_status:
    if system_mode == "🚀 Cloud Demo (Auto-Run)":
        st.markdown("<p style='text-align:right;'><span class='status-live' style='background-color: rgba(40, 167, 69, 0.2); color: #28a745; border: 1px solid #28a745;'>🟢 AUTO INGEST</span></p>", unsafe_allow_html=True)
    else:
        if st.session_state.kafka_connected:
            st.markdown("<p style='text-align:right;'><span class='status-live'>🔴 LIVE KAFKA INGEST</span></p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align:right;'><span style='color:#ffc107;'>⏳ WAITING FOR KAFKA</span></p>", unsafe_allow_html=True)

st.divider()

# ── Handle Ingress Processing ───────────────────────────────────────
new_messages = []

if system_mode == "🚀 Cloud Demo (Auto-Run)":
    # 1. Cloud simulation pipeline (Self-contained)
    model, scaler = load_app_resources()
    sim_df = load_simulation_data()
    
    if model is not None and sim_df is not None:
        if 'sim_index' not in st.session_state:
            st.session_state.sim_index = 0
            
        # Read next row of transaction
        idx = st.session_state.sim_index
        row = sim_df.iloc[idx].copy()
        
        # Advance index to loop continuously
        st.session_state.sim_index = (st.session_state.sim_index + 1) % len(sim_df)
        
        # Extract features and scale Time/Amount
        amount = float(row['Amount'])
        time_val = float(row['Time'])
        
        amount_scaled = scaler['amount_scaler'].transform([[amount]])[0][0]
        time_scaled = scaler['time_scaler'].transform([[time_val]])[0][0]
        
        # Organize features V1-V28, Amount_scaled, Time_scaled
        v_cols = [row[f"V{i}"] for i in range(1, 29)]
        features = v_cols + [amount_scaled, time_scaled]
        feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time_scaled"]
        
        # Run inference directly
        features_df = pd.DataFrame([features], columns=feature_names)
        prob = float(model.predict_proba(features_df)[0][1])
        pred = 1 if prob >= BEST_THRESHOLD else 0
        risk = "HIGH" if prob >= 0.8 else "MEDIUM" if prob >= 0.5 else "LOW"
        
        new_event = {
            "Time": time_val,
            "Amount": amount,
            "Class": int(row['Class']) if 'Class' in row else -1,
            "fraud_probability": prob,
            "prediction": pred,
            "risk_level": risk
        }
        new_messages.append(new_event)
        time.sleep(0.05) # Add soft UI breathing room
else:
    # 2. Consume from Kafka queue (Enterprise Mode)
    while not st.session_state.event_queue.empty():
        new_messages.append(st.session_state.event_queue.get())

if new_messages:
    st.session_state.transactions.extend(new_messages)
    if len(st.session_state.transactions) > st.session_state.max_buffer:
        st.session_state.transactions = st.session_state.transactions[-st.session_state.max_buffer:]

# If empty state (only in Kafka Mode when offline)
if not st.session_state.transactions:
    st.info("👋 Enterprise Mode Active! Waiting for transactions to arrive from Kafka... Switch to '🚀 Cloud Demo (Auto-Run)' in the sidebar for a fully automated cloud demonstration.")
    
    st.subheader("💡 Enterprise Pipeline Quickstart")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            **1. Boot Kafka**
            ```powershell
            docker-compose up -d
            ```
        """)
    with col2:
        st.markdown("""
            **2. Start Analytics**
            ```powershell
            python spark_fraud_detector.py
            ```
        """)
    with col3:
        st.markdown("""
            **3. Start Ingest**
            ```powershell
            python transaction_producer.py
            ```
        """)
        
    time.sleep(1)
    st.rerun()

# ── Metrics Calculations ───────────────────────────────────────
df_events = pd.DataFrame(st.session_state.transactions)

total_count = len(df_events)
fraud_cases = df_events[df_events['prediction'] == 1]
fraud_count = len(fraud_cases)
legit_count = total_count - fraud_count
fraud_percent = (fraud_count / total_count * 100) if total_count > 0 else 0.0
avg_prob = df_events['fraud_probability'].mean() if total_count > 0 else 0.0

# ── Dynamic Metric Cards ───────────────────────────────────────
st.subheader("📋 Stream Metrics (Recent Buffer)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="⚡ Ingested Volume",
        value=f"{total_count:,}",
        delta=f"+{len(new_messages)} new" if new_messages else "Streaming"
    )

with col2:
    st.metric(
        label="🚨 Flagged Fraud",
        value=f"{fraud_count}",
        delta=f"{fraud_percent:.1f}% rate",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="📊 Avg Fraud Prob",
        value=f"{avg_prob:.4f}",
        delta="Risk Index"
    )

with col4:
    st.metric(
        label="⚙️ Active Threshold",
        value=f"{BEST_THRESHOLD:.2f}",
        delta="Best F1"
    )

# ── Live Table and Threat Level ──────────────────────────────────
col_table, col_plots = st.columns([3, 2])

with col_table:
    st.subheader("⚡ Live Transaction Feed")
    
    df_display = df_events.copy()
    
    # Reverse rows so newest transactions are displayed on top
    df_display = df_display.iloc[::-1].reset_index(drop=True)
    
    df_display['Pred Class'] = df_display['prediction'].apply(lambda x: '🚨 FRAUD' if x == 1 else '✅ Legit')
    
    if 'Class' in df_display.columns:
        df_display['Ground Truth'] = df_display['Class'].apply(
            lambda x: 'FRAUD' if x == 1 else 'Legit' if x == 0 else 'Unknown'
        )
    
    cols_to_show = ['Time', 'Amount', 'fraud_probability', 'risk_level', 'Pred Class']
    if 'Ground Truth' in df_display.columns:
        cols_to_show.append('Ground Truth')
        
    df_table = df_display[cols_to_show]
    df_table = df_table.rename(columns={
        'Time': 'Time (s)',
        'Amount': 'Amount ($)',
        'fraud_probability': 'Fraud Prob',
        'risk_level': 'Risk Level'
    })
    
    def style_fraud_row(val):
        color = 'background-color: #ffcccc; color: #721c24;' if '🚨' in str(val) else ''
        return color

    st.dataframe(
        df_table.style.map(style_fraud_row, subset=['Pred Class']),
        use_container_width=True,
        height=400
    )

with col_plots:
    st.subheader("📈 Threat Distribution & Logs")
    
    tab1, tab2 = st.tabs(["Probability Distribution", "Threat Level Breakdown"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(
            df_events['fraud_probability'],
            bins=20,
            range=(0.0, 1.0),
            color='crimson' if fraud_count > 0 else 'steelblue',
            edgecolor='white',
            alpha=0.85
        )
        ax.axvline(x=BEST_THRESHOLD, color='black', linestyle='--', linewidth=1.5, label=f'Threshold ({BEST_THRESHOLD})')
        ax.set_title("Fraud Probability Distribution (Buffer)", fontsize=10)
        ax.set_xlabel("Probability", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        risk_counts = df_events['risk_level'].value_counts()
        for r in ['LOW', 'MEDIUM', 'HIGH']:
            if r not in risk_counts:
                risk_counts[r] = 0
                
        risk_counts = risk_counts.reindex(['LOW', 'MEDIUM', 'HIGH'])
        
        colors = ['#28a745', '#ffc107', '#dc3545']
        risk_counts.plot(kind='bar', color=colors, ax=ax)
        ax.set_title("Risk Category Counts", fontsize=10)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
        st.pyplot(fig)
        plt.close()

# Auto-refresh loop
# In Demo mode, we re-run faster (0.35s) for responsive fluid visual streaming.
# In Enterprise mode, we pause longer (0.6s) to reduce CPU thread cycles.
refresh_rate = 0.35 if system_mode == "🚀 Cloud Demo (Auto-Run)" else 0.6
time.sleep(refresh_rate)
st.rerun()
