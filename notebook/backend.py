from fastapi import FastAPI
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer, util
import torch

app = FastAPI()

# =========================
# LOAD EVERYTHING ONCE (CRITICAL)
# =========================
print("Loading models...")

df = pd.read_csv("vaers_sample_cleaned.csv")
embeddings = np.load("sbert_minilm_embeddings_split.npy")

vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("logreg_model.pkl")

sbert = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ Models loaded!")

# =========================
# ROUTES
# =========================

@app.get("/")
def home():
    return {"message": "ADEGuard Backend Running"}

# 🔍 Semantic Search API
@app.get("/search")
def search(query: str, k: int = 5):
    query_emb = sbert.encode([query], convert_to_tensor=True)

    scores = util.cos_sim(query_emb, embeddings)[0]
    idx = torch.topk(scores, k=k).indices.cpu().numpy()

    results = df.iloc[idx][["VAERS_ID", "symptoms_normalized"]].to_dict(orient="records")

    return {"results": results}


# ⚖️ Severity Prediction API
@app.get("/predict")
def predict(text: str):
    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0][1]
    pred = "Serious" if prob > 0.5 else "Not Serious"

    return {
        "prediction": pred,
        "probability": float(prob)
    }
