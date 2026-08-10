"""
SatGuard AI – Real‑Time Satellite Telemetry Anomaly Detection & LLM Diagnostics
Streamlit Dashboard | PGCP Big Data Analytics (CDAC) Final Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import ollama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------- Suppress trivial warnings ----------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_TOKEN"] = "1"

# ---------- Page config ----------
st.set_page_config(
    page_title="SatGuard AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Cached resource loaders ----------
@st.cache_resource
def load_model_and_data():
    ts = pd.read_csv("ts_cleaned.csv", index_col=0, parse_dates=True)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(ts.values)
    autoencoder = load_model("lstm_autoencoder.keras")
    anomaly_df = pd.read_csv("anomaly_flags.csv")
    anomaly_df['timestamp'] = pd.to_datetime(anomaly_df['timestamp'])
    anomaly_df.reset_index(drop=True, inplace=True)
    SEQ_LEN = 60
    with open("threshold.json", "r") as f:
        THRESHOLD = json.load(f)["threshold"]
    return ts, scaler, data_scaled, autoencoder, anomaly_df, SEQ_LEN, THRESHOLD

@st.cache_resource
def load_rag_pipeline():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return retriever

def generate_report(anomaly_description, retriever):
    docs = retriever.invoke(anomaly_description)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""You are a senior spacecraft fault analyst. Based on the anomaly description and the retrieved fault recovery guidelines, write a diagnostic report.

The report must contain exactly these sections:

**Anomaly Snapshot** – One sentence that repeats the timestamp, the MSE, and the threshold, and states the severity (e.g., "MSE 0.092 exceeds the 95th‑percentile threshold by 33%").

**Probable Cause** – Based on the observed behaviour and the guidelines, state the most likely cause. Use the actual numbers and the severity to justify the cause. Vary the wording for each report; do not copy a generic phrase.

**Recommended Actions** – A bullet list of recovery steps pulled directly from the guidelines. If the anomaly is very severe, recommend immediate action; otherwise, standard checks.

**Confidence Note** – One sentence: "High confidence" if the MSE is >20% above threshold, else "Moderate confidence".

Keep the report between 120 and 160 words. Use the specific numbers from the description.

Anomaly description:
{anomaly_description}

Guidelines:
{context}

Diagnostic Report:"""

    response = ollama.chat(
        model="llama3:8b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ---------- Load all resources ----------
ts, scaler, data_scaled, autoencoder, anomaly_df, SEQ_LEN, THRESHOLD = load_model_and_data()
retriever = load_rag_pipeline()

# ---------- Header ----------
st.title("🛰️ SatGuard AI")
st.markdown(
    "**Real‑Time Satellite Telemetry Anomaly Detection & "
    "LLM‑Generated Fault Diagnostic Reports**"
)
st.markdown("---")

# ---------- KPI Metrics (styled as cards) ----------
st.markdown("""
<style>
.kpi-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    text-align: center;
    margin: 5px;
}
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    color: #1f77b4;
}
.kpi-label {
    font-size: 14px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

cols = st.columns(4)
metrics = [
    ("Total Anomalies Detected", len(anomaly_df), None),
    ("Avg MSE", f"{anomaly_df['mse'].mean():.4f}", None),
    ("Max MSE", f"{anomaly_df['mse'].max():.4f}", None),
    ("Threshold (95th %ile)", f"{THRESHOLD:.5f}", None)
]
for col, (label, value, _) in zip(cols, metrics):
    with col:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["📋 Anomaly Inspection", "⚡ Live Stream Demo"])

# ======================== TAB 1: INSPECTION ========================
with tab1:
    col_sidebar, col_main = st.columns([1, 3])
    with col_sidebar:
        st.subheader("📌 Select Anomaly")
        anomaly_idx = st.selectbox(
            "Choose an anomaly to inspect:",
            options=anomaly_df.index.tolist(),
            format_func=lambda x: f"#{x} — {anomaly_df.loc[x, 'timestamp']}"
        )
        selected = anomaly_df.loc[anomaly_idx]
        st.markdown("---")
        st.markdown("**Anomaly Details**")
        st.write(f"📅 Timestamp: {selected['timestamp']}")
        st.write(f"📊 MSE: {selected['mse']:.5f}")
        above = selected['mse'] > THRESHOLD
        st.write(f"🔺 Above Threshold: {'✅ Yes' if above else '❌ No'}")

    with col_main:
        st.subheader("📈 Telemetry Snapshot")
        window_minutes = st.slider("Time window (± minutes):", 1, 30, 5, key="win")
        start = selected['timestamp'] - pd.Timedelta(minutes=window_minutes)
        end   = selected['timestamp'] + pd.Timedelta(minutes=window_minutes)
        ts_window = ts.loc[start:end]

        all_channels = ts.columns.tolist()
        default_channels = all_channels[:3]
        selected_channels = st.multiselect(
            "Channels to display:", all_channels, default=default_channels
        )

        if not ts_window.empty and selected_channels:
            fig, ax = plt.subplots(figsize=(10, 4))
            for ch in selected_channels:
                if ch in ts_window.columns:
                    ax.plot(ts_window.index, ts_window[ch], label=ch, alpha=0.8, linewidth=0.8)
            ax.axvline(selected['timestamp'], color='red', linestyle='--', linewidth=2, label='Anomaly')
            ax.legend(loc='upper right', fontsize='small')
            ax.set_xlabel("Time")
            ax.set_ylabel("Raw Sensor Value")
            st.pyplot(fig)
        else:
            st.info("No data in the selected window or no channels chosen.")

        st.subheader("🤖 AI‑Generated Diagnostic Report")
        if st.button("Generate Report", type="primary", key="report_btn"):
            mse = selected['mse']
            pct = ((mse - THRESHOLD) / THRESHOLD) * 100.0
            severity = "very severe" if pct > 20 else "moderate"
            desc = (
                f"Anomaly at {selected['timestamp']}.\n"
                f"Reconstruction MSE: {mse:.5f} ({pct:.1f}% above the 95th‑percentile threshold of {THRESHOLD:.5f}). "
                f"This deviation is {severity} and involves one or more of the 9 OPS‑SAT sensor channels."
            )
            with st.spinner("Retrieving knowledge and consulting the LLM... (may take up to 60 seconds)"):
                report = generate_report(desc, retriever)
            st.markdown(report)
        else:
            st.info("Click the button above to generate a diagnostic report for the selected anomaly.")

# ======================== TAB 2: LIVE STREAM DEMO ========================
with tab2:
    st.subheader("⚡ Real‑Time Streaming Simulation")
    st.markdown(
        "Replay telemetry window‑by‑window, detect anomalies, "
        "and generate LLM reports on‑the‑fly."
    )

    # ---------- Pre‑compute the anomaly cluster start position ----------
    first_anomaly_time = anomaly_df.iloc[0]['timestamp']
    end_idx = ts.index.get_indexer([first_anomaly_time], method='nearest')[0]
    window_start_idx = max(0, end_idx - (SEQ_LEN - 1))
    cluster_start_seq_idx = max(0, window_start_idx - 3)
    total_sequences = len(data_scaled) - SEQ_LEN + 1
    cluster_start_seq_idx = min(cluster_start_seq_idx, total_sequences - 1)

    col_ctrl, col_status = st.columns([1, 1])
    with col_ctrl:
        max_windows = st.slider(
            "Number of windows to replay:",
            min_value=5, max_value=100, value=10, step=5,
            key="stream_slider"
        )

        time_range = st.selectbox(
            "Replay period:",
            options=["Early mission (Jan 2022)", "Late mission (Jun 2022)",
                     "Cluster (Jun 2 anomalies)", "Custom start index"]
        )

        if time_range == "Early mission (Jan 2022)":
            start_offset = 0
        elif time_range == "Late mission (Jun 2022)":
            split_idx = int(0.8 * total_sequences)
            start_offset = split_idx
        elif time_range == "Cluster (Jun 2 anomalies)":
            start_offset = cluster_start_seq_idx
        else:
            max_start = max(0, total_sequences - max_windows)
            start_offset = st.slider(
                "Start window index:",
                min_value=0, max_value=max_start, value=0, step=10
            )

        start_btn = st.button("▶️ Start Streaming")

    with col_status:
        status_placeholder = st.empty()

    if start_btn:
        # Prepare data range
        start_idx = max(0, min(start_offset, total_sequences - max_windows))
        data_subset = data_scaled[start_idx : start_idx + max_windows + SEQ_LEN]
        timestamps_subset = ts.index[start_idx : start_idx + max_windows + SEQ_LEN]
        total_windows = min(max_windows, len(data_subset) - SEQ_LEN + 1)

        # Show info about expected anomalies
        known_in_range = anomaly_df[
            (anomaly_df['timestamp'] >= timestamps_subset[SEQ_LEN-1]) &
            (anomaly_df['timestamp'] <= timestamps_subset[-1])
        ]
        status_placeholder.info(
            f"ℹ️ {len(known_in_range)} known anomalies in this range. "
            f"First anomaly expected near {first_anomaly_time}."
        )

        progress_bar = st.progress(0)
        output_area = st.empty()

        anomaly_counter = 0
        stream_output = ""

        # Process all windows sequentially
        for i in range(total_windows):
            window = data_subset[i:i+SEQ_LEN]
            timestamp = timestamps_subset[i+SEQ_LEN-1]

            # Anomaly check
            X = window.reshape(1, SEQ_LEN, 9)
            X_pred = autoencoder.predict(X, verbose=0)
            mse = np.mean(np.square(X - X_pred))
            flag = mse > THRESHOLD

            if flag:
                anomaly_counter += 1
                pct = ((mse - THRESHOLD) / THRESHOLD) * 100.0
                severity = "very severe" if pct > 20 else "moderate"
                desc = (
                    f"Anomaly at {timestamp}.\n"
                    f"MSE: {mse:.5f} ({pct:.1f}% above threshold {THRESHOLD:.5f}). "
                    f"Deviation is {severity}."
                )

                with st.spinner(f"Generating report for anomaly at {timestamp}..."):
                    report = generate_report(desc, retriever)
                output = (
                    f"### 🚨 Anomaly #{anomaly_counter}\n"
                    f"**Time:** {timestamp}\n"
                    f"**MSE:** {mse:.5f}\n\n"
                    f"{report}\n"
                    f"---\n"
                )
                stream_output += output

            # Update progress bar and show reports so far
            progress_bar.progress((i+1) / total_windows)
            output_area.markdown(stream_output)

            # Small delay for realism
            time.sleep(0.05)

        status_placeholder.success(
            f"Streaming completed. {anomaly_counter} anomalies detected."
        )

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: grey;'>"
    "SatGuard AI | <b>Marshenil Mitra</b> | "
    "PGCP Big Data Analytics, CDAC | Final Capstone Project<br>"
    "All components run locally on CPU using open‑source tools."
    "</div>",
    unsafe_allow_html=True
)