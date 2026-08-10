# rag_evaluation.py – Lightweight RAG Retrieval Relevance Check (fast, fixed)
import json
import random
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma

# ---------- 1. Fast embedding wrapper for LangChain ----------
class FastEmbed:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(f"sentence-transformers/{model_name}")
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

# ---------- 2. Loading the vector store with the fast embedder ----------
print("Loading vector store...")
embedder = FastEmbed()
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedder
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ---------- 3. Loading real anomaly sample ----------
df = pd.read_csv("anomaly_flags.csv")
sample_size = min(30, len(df))
random.seed(42)
sample = df.sample(n=sample_size, random_state=42)

# ---------- 4. Relevance check ----------
def is_relevant(chunks):
    keywords = ["magnetometer", "power", "cycling", "drop", "spike", "latch-up", "solder"]
    combined = " ".join(chunks).lower()
    #return any(kw in combined for kw in keywords)
    matched = [kw for kw in keywords if kw in combined]
    return len(matched) >= 2

# ---------- 5. Evaluating ----------
relevant_count = 0
for _, row in sample.iterrows():
    desc = f"Anomaly at {row['timestamp']}. MSE: {row['mse']:.5f}. Deviation in one or more sensor channels."
    docs = retriever.invoke(desc)
    chunks = [doc.page_content for doc in docs]
    if is_relevant(chunks):
        relevant_count += 1

# ---------- 6. Saving results ----------
relevance_rate = relevant_count / sample_size
results = {
    "sample_size": sample_size,
    "relevant_retrievals": relevant_count,
    "relevance_rate": round(relevance_rate, 3),
    "method": "Automated keyword-based relevance check on retrieved chunks.",
    "limitations": "Preliminary baseline. Future work includes human evaluation and answer correctness metrics."
}
with open("rag_evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved rag_evaluation_results.json – Relevance Rate: {relevance_rate:.2%}")