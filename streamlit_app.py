import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# ── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open('fraud_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# ── Load Scalers ─────────────────────────────────────────────
# FIX: Load the training scalers so inference uses the SAME scaling as training.
# Previously the app created a brand-new StandardScaler() and fit it on the
# uploaded data — a completely different distribution from training.
@st.cache_resource
def load_scalers():
    try:
        with open('scaler.pkl', 'rb') as f:
            return pickle.load(f)   # returns {'amount_scaler': ..., 'time_scaler': ...}
    except Exception as e:
        st.warning(f"⚠️ Could not load scaler.pkl: {e}. Falling back to fresh scaling.")
        return None

# ── Load Threshold from Config ────────────────────────────────
# FIX: Read the threshold computed by the notebook (best F1 on validation set)
# instead of hardcoding 0.89 which only worked on the internal test split.
def load_config():
    try:
        with open('model_config.json') as f:
            return json.load(f)
    except Exception:
        return {'best_threshold': 0.5}

model   = load_model()
scalers = load_scalers()
config  = load_config()
DEFAULT_THRESHOLD = float(config.get('best_threshold', 0.5))

# Stop execution if model failed to load
if model is None:
    st.stop()

# ── Header ───────────────────────────────────────────────────
st.title("🔍 Credit Card Fraud Detection System")
st.markdown("**XGBoost + SHAP Explainability** | Trained on 284,807 transactions | ROC-AUC: 0.9769")
st.divider()

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.01, max_value=0.99,
    value=DEFAULT_THRESHOLD, step=0.01,
    help="Loaded from model_config.json (best F1 threshold). Higher = fewer false alarms, may miss some fraud."
)
show_shap   = st.sidebar.checkbox("Show SHAP Explanations", value=True)
max_explain = st.sidebar.slider("Max transactions to explain", 1, 20, 4)

st.sidebar.divider()
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.metric("ROC-AUC", "0.9769")
st.sidebar.metric("Precision (Fraud)", "0.87")
st.sidebar.metric("Recall (Fraud)", "0.83")
st.sidebar.metric("F1 Score", "0.85")

# ── File Upload ──────────────────────────────────────────────
st.subheader("📂 Upload Transactions CSV")
st.markdown("Upload a CSV with columns V1-V28, Amount, Time (same format as creditcard.csv)")

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Separate ground truth if present
    true_labels = None
    if 'Class' in df.columns:
        true_labels = df['Class'].values
        df = df.drop('Class', axis=1)

    # ── Scaling ──────────────────────────────────────────────
    # FIX: Use the SAVED training scalers (transform only — never fit on new data).
    # If the CSV already has Amount_scaled / Time_scaled (pre-processed format),
    # skip this block entirely.
    if 'Amount' in df.columns and 'Time' in df.columns:
        if scalers is not None:
            # Correct path: use training distribution statistics
            df['Amount_scaled'] = scalers['amount_scaler'].transform(df[['Amount']])
            df['Time_scaled']   = scalers['time_scaler'].transform(df[['Time']])
        else:
            # Fallback (scaler.pkl missing) — warn user, results may be less accurate
            from sklearn.preprocessing import StandardScaler
            st.warning("⚠️ scaler.pkl not found — using fallback scaling. Re-run the notebook to generate it.")
            df['Amount_scaled'] = StandardScaler().fit_transform(df[['Amount']])
            df['Time_scaled']   = StandardScaler().fit_transform(df[['Time']])
        df = df.drop(['Amount', 'Time'], axis=1)

    st.success(f"✅ Loaded {len(df)} transactions")

    # ── Predictions ──────────────────────────────────────────
    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)

    fraud_count = preds.sum()
    legit_count = len(preds) - fraud_count

    # ── Summary Metrics ──────────────────────────────────────
    st.divider()
    st.subheader("📊 Prediction Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", len(df))
    col2.metric("🚨 Flagged as Fraud", int(fraud_count),
                delta=f"{fraud_count/len(df)*100:.1f}%", delta_color="inverse")
    col3.metric("✅ Legitimate", int(legit_count))
    col4.metric("Threshold Used", f"{threshold:.2f}")

    # ── Results Table ─────────────────────────────────────────
    st.divider()
    st.subheader("📋 Transaction Results")

    results_df = pd.DataFrame({
        'Transaction #':    range(1, len(df)+1),
        'Fraud Probability': [f"{p:.4f}" for p in probs],
        'Prediction':        ['🚨 FRAUD' if p == 1 else '✅ Legit' for p in preds],
        'Risk Level':        ['HIGH' if p >= 0.8 else 'MEDIUM' if p >= 0.5 else 'LOW' for p in probs]
    })

    if true_labels is not None:
        results_df['Actual'] = ['FRAUD' if l == 1 else 'Legit' for l in true_labels]
        correct = (preds == true_labels).sum()
        st.info(f"Ground truth available — Correct predictions: {correct}/{len(df)}")

    def highlight_fraud(row):
        if '🚨' in row['Prediction']:
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)

    st.dataframe(
        results_df.style.apply(highlight_fraud, axis=1),
        use_container_width=True
    )

    # ── Probability Distribution ──────────────────────────────
    st.divider()
    st.subheader("📈 Fraud Probability Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(probs[preds==0], bins=50, alpha=0.7,
            color='steelblue', label='Predicted Legit')
    ax.hist(probs[preds==1], bins=50, alpha=0.7,
            color='crimson', label='Predicted Fraud')
    ax.axvline(x=threshold, color='black', linestyle='--',
               linewidth=2, label=f'Threshold = {threshold:.2f}')
    ax.set_xlabel('Fraud Probability')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Fraud Probabilities')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # ── SHAP Explanations ─────────────────────────────────────
    if show_shap and fraud_count > 0:
        st.divider()
        st.subheader("🧠 SHAP Explanations — Why These Were Flagged")

        explainer    = shap.TreeExplainer(model)
        fraud_indices = np.where(preds == 1)[0][:max_explain]

        for i, idx in enumerate(fraud_indices):
            with st.expander(f"🚨 Transaction #{idx+1} — Probability: {probs[idx]:.4f}"):
                shap_vals = explainer.shap_values(df.iloc[[idx]])

                fig, ax = plt.subplots(figsize=(10, 5))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_vals[0],
                        base_values=explainer.expected_value,
                        data=df.iloc[idx].values,
                        feature_names=df.columns.tolist()
                    ),
                    show=False
                )
                st.pyplot(fig)
                plt.close()

    elif show_shap and fraud_count == 0:
        st.info("No fraud detected in this batch — try lowering the threshold.")

else:
    # ── Landing State ─────────────────────────────────────────
    st.info("👆 Upload a CSV file to begin fraud analysis")

    st.subheader("📌 How to use")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload CSV**\nUpload transaction data in creditcard.csv format (V1-V28, Amount, Time, optional Class)")
    with col2:
        st.markdown("**2. Adjust Threshold**\nDefault is the best F1 threshold from training. Use sidebar to tune precision vs recall.")
    with col3:
        st.markdown("**3. Review Results**\nSee flagged transactions with SHAP explanations")
