#!/usr/bin/env python3
"""
SatGuard AI – Kafka Streaming Proof-of-Concept
Replays telemetry windows into a Kafka topic and runs anomaly detection
using the trained LSTM-Autoencoder.
"""

import json
import time
import threading
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from kafka import KafkaProducer, KafkaConsumer

# ---------- Configuration ----------
TOPIC = "satellite_telemetry"
BOOTSTRAP_SERVERS = "localhost:9092"
SEQ_LEN = 60
MAX_WINDOWS = 200          # how many windows to stream (demo length)
SLEEP_MS = 0.05            # tiny delay to mimic real-time

# ---------- Load model and data ----------
print("Loading model and data...")
ts = pd.read_csv("ts_cleaned.csv", index_col=0, parse_dates=True)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(ts.values)

autoencoder = load_model("lstm_autoencoder.keras")
with open("threshold.json", "r") as f:
    THRESHOLD = json.load(f)["threshold"]

# ---------- Producer ----------
def produce_data():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )
    windows = min(MAX_WINDOWS, len(data_scaled) - SEQ_LEN)
    print(f"Producer: sending {windows} windows to topic '{TOPIC}'...")
    for i in range(windows):
        window = data_scaled[i:i+SEQ_LEN].tolist()
        timestamp = str(ts.index[i+SEQ_LEN-1])
        message = {"window_index": i, "window": window, "timestamp": timestamp}
        producer.send(TOPIC, message)
        time.sleep(SLEEP_MS)
    producer.flush()
    print("Producer: finished sending windows.")

# ---------- Consumer ----------
def consume_data():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8"))
    )
    print("Consumer: waiting for messages...")
    for msg in consumer:
        data = msg.value
        window = np.array(data["window"])
        X = window.reshape(1, SEQ_LEN, 9)
        X_pred = autoencoder.predict(X, verbose=0)
        mse = np.mean(np.square(X - X_pred))
        if mse > THRESHOLD:
            print(f"🚨 Anomaly at {data['timestamp']} | MSE: {mse:.5f}")
        else:
            print(f"   Normal at {data['timestamp']} | MSE: {mse:.5f}")

# ---------- Run both in parallel ----------
if __name__ == "__main__":
    producer_thread = threading.Thread(target=produce_data)
    consumer_thread = threading.Thread(target=consume_data)

    producer_thread.start()
    time.sleep(3)          # wait for a few windows to be produced
    consumer_thread.start()

    producer_thread.join()
    # consumer runs until you press Ctrl+C
    print("Demo running. Press Ctrl+C to stop.")