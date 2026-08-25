markdown
# 🛰️ SatGuard AI

**Real‑Time Satellite Telemetry Anomaly Detection Pipeline with Deep Learning, Multi‑Subsystem Feature Fusion, and LLM‑Generated Fault Diagnostic Reports**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM--Autoencoder-orange)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerised-green)](https://www.docker.com/)
[![Kafka](https://img.shields.io/badge/Kafka-Streaming-red)](https://kafka.apache.org/)
[![GitHub](https://img.shields.io/badge/Status-Completed-brightgreen)](https://github.com/marshenilmitra/SatGuard-AI)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Demo](#demo)
- [Evaluation](#evaluation)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Docker Deployment](#docker-deployment)
- [Kafka Streaming Proof‑of‑Concept](#kafka-streaming-proof‑of‑concept)
- [Documentation & Resources](#documentation--resources)
- [Limitations & Future Work](#limitations--future-work)
- [License](#license)
- [Contact](#contact)

---

## 🧭 Overview

SatGuard AI ingests raw multi‑subsystem satellite telemetry, fuses 9 sensor channels, and flags abnormal behaviour in real time. When an anomaly fires, the system retrieves relevant fault‑recovery guidelines from a local knowledge base and uses a locally hosted LLM to explain the fault and recommend actions. The entire pipeline is packaged into an interactive Streamlit dashboard, containerised with Docker, and extended with a Kafka streaming proof‑of‑concept.

---

## ✨ Key Features

- **Real data, messy data** – 300K+ records from ESA’s OPS‑SAT CubeSat, 9 sensor channels.
- **Multi‑subsystem feature fusion** – raw channels + rolling statistics (45 features for baseline models).
- **Deep learning anomaly detection** – LSTM‑Autoencoder trained on 60‑timestep windows; threshold at 95th percentile of reconstruction error.
- **Explainable diagnostics** – RAG pipeline (ChromaDB + Llama 3) produces structured reports with probable cause, actions, and confidence.
- **Fully offline LLM** – no internet, no API costs, secure.
- **Interactive dashboard** – Streamlit with anomaly inspection and live‑stream replay tabs.
- **Containerised deployment** – Dockerfile for reproducible builds.
- **Streaming awareness** – Apache Kafka producer/consumer POC for scalable ingestion.

---

## 🧠 System Architecture

![SatGuard AI Architecture](images/architecture.png)

The pipeline consists of four layers:

1. **Data Ingestion & Preprocessing** – CSV telemetry → pivot → fill → scale → windowing.
2. **Anomaly Detection** – LSTM‑Autoencoder computes reconstruction error; above‑threshold windows are anomalies.
3. **Explainability & Diagnostics** – RAG retrieves fault guidelines and LLM writes a report.
4. **Presentation & Deployment** – Streamlit dashboard, Docker container, Kafka POC.

Detailed RAG flow:

![RAG Pipeline](images/rag_pipeline.png)

---

## 🎬 Demo

### Quick Dashboard Demo (auto‑playing GIF)
![SatGuard AI Dashboard Demo](demo/demo.gif)

### Full Walkthrough (YouTube)
▶️ [Watch the full walkthrough](https://youtu.be/mchuxfH_WaY)

---

## 📊 Evaluation

**Anomaly detector baseline (validation set):**

| Metric | Value |
|--------|-------|
| Precision | **1.000** |
| Recall | **0.024** |
| F1‑score | **0.046** |

This is a **high‑precision, low‑recall baseline**—the system only fires when it is extremely confident, so there are **zero false alarms**. Recall improvement is planned via threshold tuning and refined window labelling.

**RAG retrieval baseline:**  
Automated keyword‑based check on 30 random anomalies gave **100% relevance** (sanity check, not full human evaluation).

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| ML/DL | TensorFlow/Keras, Scikit‑learn |
| RAG / Vector DB | LangChain, ChromaDB, sentence‑transformers |
| LLM | Ollama + Llama 3 (8B, local) |
| Dashboard | Streamlit |
| Container | Docker |
| Streaming POC | Apache Kafka, kafka‑python |
| Version Control | Git, GitHub |

---

## 📁 Repository Structure
SatGuard-AI/
├── app.py # Streamlit dashboard
├── satguard_eda.ipynb # EDA + baseline Isolation Forest
├── lstm_autoencoder.ipynb # LSTM‑Autoencoder training + evaluation
├── rag_llm_diagnostics.ipynb # RAG pipeline + report generation
├── streaming_simulation.ipynb # Real‑time file‑replay simulation
├── kafka_streaming_demo.py # Kafka producer/consumer POC
├── docker-compose.yml # Kafka + Zookeeper
├── Dockerfile # Dashboard container
├── requirements.txt # Python dependencies
├── threshold.json # Locked anomaly threshold
├── evaluation_metrics.json # Precision/Recall/F1
├── rag_evaluation_results.json # RAG relevance baseline
├── ts_cleaned.csv # Cleaned pivoted telemetry
├── anomaly_flags.csv # Validation anomalies
├── lstm_autoencoder.keras # Trained model
├── chroma_db/ # Persisted vector store
├── knowledge_base/ # Fault recovery guidelines (4 txt files)
├── images/ # Architecture diagrams
├── docs/ # PPT + final report
└── demo/ # Demo GIF + short video

text

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/marshenilmitra/SatGuard-AI.git
cd SatGuard-AI
2. Create Python 3.11 environment (recommended)
bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt
3. Install Ollama and pull Llama 3
Download Ollama from ollama.com

Pull the model:

bash
ollama pull llama3:8b
4. Run the Streamlit dashboard
bash
streamlit run app.py
Open http://localhost:8501.

🐳 Docker Deployment
Build and run the container (mount data files for full functionality):

bash
docker build -t satguard .
docker run -p 8501:8501 \
  -v ${PWD}/ts_cleaned.csv:/app/ts_cleaned.csv \
  -v ${PWD}/lstm_autoencoder.keras:/app/lstm_autoencoder.keras \
  -v ${PWD}/chroma_db:/app/chroma_db \
  -v ${PWD}/threshold.json:/app/threshold.json \
  -v ${PWD}/anomaly_flags.csv:/app/anomaly_flags.csv \
  -v ${PWD}/evaluation_metrics.json:/app/evaluation_metrics.json \
  satguard
⚡ Kafka Streaming Proof‑of‑Concept
Start Kafka & Zookeeper:

bash
docker compose up -d
Run the demo:

bash
python kafka_streaming_demo.py
You will see real‑time telemetry windows and anomaly alerts printed in the console. Stop services:

bash
docker compose down
📚 Documentation & Resources
Project Report: docs/SatGuard_AI_Project_Report.pdf

Presentation: docs/SatGuard_AI_Presentation.pdf

🔮 Limitations & Future Work
Recall improvement – lower threshold / refine window labelling to catch more anomalies.

True real‑time – integrate Kafka directly into the dashboard.

Expand knowledge base – include more fault types and subsystem guidelines.

Advanced RAG evaluation – human relevance judgments, LLM‑as‑a‑judge.

Feature‑level explainability – SHAP or attention visualisation for sensor attribution.

📝 License
This project is open‑source under the MIT License. See LICENSE file for details.

👤 Contact
Marshenil Mitra
PGCP Big Data Analytics, CDAC
GitHub | LinkedIn