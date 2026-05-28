import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import queue
import threading
import os
import matplotlib.pyplot as plt
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable

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
""", unsafe_allowed_html=True)

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

# Kafka consumer configuration
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
ALERTS_TOPIC = os.getenv("ALERTS_TOPIC", "fraud_alerts")

# Background thread for consuming from Kafka
def kafka_consumer_thread(broker_addr, topic_name, q):
    print("Background consumer thread started.")
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

# Start consumer thread once
if 'consumer_started' not in st.session_state:
    st.session_state.consumer_started = True
    t = threading.Thread(
        target=kafka_consumer_thread,
        args=(KAFKA_BROKER, ALERTS_TOPIC, st.session_state.event_queue),
        daemon=True
    )
    t.start()

# ── Sidebar Configurations ─────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/nolan/128/security-shield.png", width=70)
st.sidebar.markdown("<h2 style='margin-top:0;'>🛡️ FRAUD GUARD</h2>", unsafe_allowed_html=True)
st.sidebar.divider()

# Status indicator
if st.session_state.kafka_connected:
    st.sidebar.markdown("""
        <div class="status-live">
            <span style="height: 8px; width: 8px; background-color: #dc3545; border-radius: 50%; display: inline-block;"></span>
            LIVE FEED ACTIVE
        </div>
    """, unsafe_allowed_html=True)
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
    st.markdown('<div class="title-gradient">Real-Time Fraud Guard</div>', unsafe_allowed_html=True)
    st.markdown("##### Real-Time Credit Card Fraud Detection Powered by Apache Kafka & Spark Structured Streaming")
with col_status:
    if st.session_state.kafka_connected:
        st.markdown("<p style='text-align:right;'><span class='status-live'>🔴 LIVE KAFKA INGEST</span></p>", unsafe_allowed_html=True)
    else:
        st.markdown("<p style='text-align:right;'><span style='color:#ffc107;'>⏳ WAITING FOR KAFKA</span></p>", unsafe_allowed_html=True)

st.divider()

# Ingest new events from background thread
new_messages = []
while not st.session_state.event_queue.empty():
    new_messages.append(st.session_state.event_queue.get())

if new_messages:
    # Prepend new events so recent is top, or append
    st.session_state.transactions.extend(new_messages)
    # Trim to fit max buffer limit
    if len(st.session_state.transactions) > st.session_state.max_buffer:
        st.session_state.transactions = st.session_state.transactions[-st.session_state.max_buffer:]

# If empty state
if not st.session_state.transactions:
    st.info("👋 Live Feed Connected! Waiting for simulated transactions to arrive... Run your `transaction_producer.py` and `spark_fraud_detector.py` to stream events.")
    
    st.subheader("💡 System Startup Quickguide")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            **1. Boot Infrastructure**
            ```powershell
            docker-compose up -d
            ```
        """)
    with col2:
        st.markdown("""
            **2. Start Streaming Engine**
            ```powershell
            python spark_fraud_detector.py
            ```
        """)
    with col3:
        st.markdown("""
            **3. Start Stream Simulation**
            ```powershell
            python transaction_producer.py
            ```
        """)
        
    # Auto-refresh loop to wait for data
    time.sleep(1)
    st.rerun()

# Data is available
df_events = pd.DataFrame(st.session_state.transactions)

# Metric Calculations
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
    
    # Format table for representation
    df_display = df_events.copy()
    
    # Reverse rows so newest transactions are displayed on top
    df_display = df_display.iloc[::-1].reset_index(drop=True)
    
    # Add human readable styling columns
    df_display['Pred Class'] = df_display['prediction'].apply(lambda x: '🚨 FRAUD' if x == 1 else '✅ Legit')
    
    # Handle optional Class column for ground truth monitoring
    if 'Class' in df_display.columns:
        df_display['Ground Truth'] = df_display['Class'].apply(
            lambda x: 'FRAUD' if x == 1 else 'Legit' if x == 0 else 'Unknown'
        )
    
    # Custom display subsets
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
    
    # Color highlight function for Streamlit table rows
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
    
    # Generate live probability histogram and risk charts
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
        # Ensure all columns exist for layout consistency
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

# Auto-refresh mechanism
# This re-runs the page every 0.6 seconds to pull new data from background queue
time.sleep(0.6)
st.rerun()
