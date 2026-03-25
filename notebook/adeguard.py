
import streamlit as st
import sqlite3
import hashlib

# =========================
# PAGE CONFIG (Hard UI Anchor)
# =========================
st.set_page_config(page_title="ADEGuard Explorer", layout="wide")

# =========================
# DATABASE SETUP
# =========================
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# =========================
# AUTH HELPERS
# =========================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES (?, ?)",
                  (username, hash_password(password)))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def login_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?",
              (username, hash_password(password)))
    data = c.fetchone()
    conn.close()
    return data is not None

# =========================
# SESSION STATE
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# =========================
# AUTH SCREEN
# =========================
if not st.session_state.logged_in:

    st.markdown("""
        <style>
        body {
            background: linear-gradient(135deg, #1f1c2c, #928dab);
        }
        .auth-box {
            width: 400px;
            margin: auto;
            padding: 40px;
            border-radius: 20px;
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(15px);
            box-shadow: 0 0 25px rgba(0,0,0,0.4);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center;'>🔐 ADEGuard Explorer</h1>", unsafe_allow_html=True)

    menu = st.radio("", ["Login", "Register"], horizontal=True)

    with st.container():
        st.markdown("<div class='auth-box'>", unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if menu == "Register":
            if st.button("Create Account"):
                if username and password:
                    if register_user(username, password):
                        st.success("Account created! Please login.")
                    else:
                        st.error("Username already exists")
                else:
                    st.warning("Enter username and password")

        else:
            if st.button("Login"):
                if login_user(username, password):
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()

# =========================
# LOGOUT BUTTON
# =========================
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# =========================
# CORE APP STARTS HERE
# =========================

st.title("🚀 ADEGuard Explorer")

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util
import joblib
import ast

# =========================
# Cached loaders
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("ADEGuard/notebook/vaers_sample_cleaned.csv")
    embeddings = np.load("ADEGuard/notebook/sbert_minilm_embeddings_split.npy")
    return df, embeddings

@st.cache_resource
def load_models():
    vectorizer = joblib.load("ADEGuard/notebook/tfidf_vectorizer.pkl")
    model = joblib.load("ADEGuard/notebook/logreg_model.pkl")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return vectorizer, model, sbert

@st.cache_data
def compute_pca(embeddings):
    return PCA(n_components=2).fit_transform(embeddings)

# =========================
# Load core assets safely
# =========================
try:
    df, embeddings = load_data()
    vectorizer, model, sbert = load_models()
    emb2d_all = compute_pca(embeddings)
except Exception as e:
    st.error("❌ Startup failure")
    st.exception(e)
    st.stop()

# =========================
# Helpers
# =========================
def safe_list_to_str(x):
    try:
        return ", ".join(ast.literal_eval(x)) if isinstance(x, str) else str(x)
    except:
        return str(x)

df["VAX_TYPE_STR"] = df["VAX_TYPE"].apply(safe_list_to_str)
df["VAX_MANU_STR"] = df["VAX_MANU"].apply(safe_list_to_str)

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Dataset Preview", "PCA Visualization", "Semantic Search", "Severity Prediction"]
)

# =========================
# Pages
# =========================

if page == "Home":
    st.header("Home")
    col1, col2 = st.columns(2)
    col1.metric("Dataset Rows", df.shape[0])
    col2.metric("Embedding Dimensions", embeddings.shape[1])

elif page == "Dataset Preview":
    st.header("Dataset Preview")
    rows = st.slider("Rows:", 5, 50, 10)
    st.dataframe(df.head(rows), use_container_width=True)

elif page == "PCA Visualization":
    st.header("PCA Visualization")
    n = st.slider("Points:", 500, min(10000, len(df)), 2000)

    fig = px.scatter(
        x=emb2d_all[:n, 0],
        y=emb2d_all[:n, 1],
        color=df["AGE_YRS"].head(n),
        opacity=0.6,
        color_continuous_scale="Viridis",
        labels={"x": "PCA-1", "y": "PCA-2"}
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Semantic Search":
    st.header("Semantic Search")
    query = st.text_input("Search symptoms")

    if query:
        import torch
        query_emb = sbert.encode([query], convert_to_tensor=True)
        scores = util.cos_sim(query_emb, embeddings)[0]
        k = st.slider("Top K", 1, 20, 5)
        idx = torch.topk(scores, k=k).indices.cpu().numpy()

        for _, row in df.iloc[idx].iterrows():
            with st.expander(f"VAERS {row['VAERS_ID']}"):
                st.write(row["symptoms_normalized"])

elif page == "Severity Prediction":
    st.header("Severity Prediction")

    text = st.text_area("Enter symptoms")
    use_llm = st.checkbox("Explain with Mistral (Ollama)")

    if st.button("Predict"):
        if not text.strip():
            st.warning("Enter symptoms")
            st.stop()

        X = vectorizer.transform([text])
        prob = model.predict_proba(X)[0][1]
        pred = "Serious" if prob > 0.5 else "Not Serious"

        st.success(pred)
        st.write(f"Probability (Serious): {prob:.2f}")

        if use_llm:
            with st.spinner("Calling Mistral…"):
                try:
                    import ollama

                    prompt = f"""
Explain this prediction briefly.

Symptoms: {text}
Prediction: {pred}
Probability: {prob:.2f}

No medical advice.
"""
                    r = ollama.chat(
                        model="mistral",
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": 0.2}
                    )

                    st.info(r["message"]["content"])

                except Exception as e:
                    st.warning("Mistral unavailable")
                    st.caption(str(e))
