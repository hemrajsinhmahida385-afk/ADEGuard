import streamlit as st
import requests
import sqlite3
import hashlib
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
# CONFIG
# =========================
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="ADEGuard Explorer", layout="wide")

# =========================
# DATABASE (AUTH)
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
# AUTH FUNCTIONS
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
# AUTH UI
# =========================
if not st.session_state.logged_in:

    st.title("🔐 ADEGuard Explorer")

    menu = st.radio("", ["Login", "Register"], horizontal=True)

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
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.stop()

# =========================
# LOGOUT
# =========================
st.sidebar.success("Logged in ✅")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# =========================
# MAIN APP
# =========================

st.title("🚀 ADEGuard Explorer")

# =========================
# LOAD LIGHT DATA
# =========================
@st.cache_data
def load_preview():
    return pd.read_csv("ADEGuard/notebook/vaers_sample_cleaned.csv")

@st.cache_data
def load_pca():
    import os
    from sklearn.decomposition import PCA

    path = "ADEGuard/notebook/pca.npy"

    if os.path.exists(path):
        return np.load(path)
    else:
        embeddings = np.load("ADEGuard/notebook/sbert_minilm_embeddings_split.npy")
        emb2d = PCA(n_components=2).fit_transform(embeddings)
        np.save(path, emb2d)
        return emb2d

df = load_preview()
emb2d = load_pca()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Dataset Preview", "PCA Visualization", "Semantic Search", "Severity Prediction"]
)

# =========================
# PAGES
# =========================

# 🏠 HOME
if page == "Home":
    st.header("Home")
    col1, col2 = st.columns(2)
    col1.metric("Dataset Rows (Preview)", df.shape[0])
    col2.metric("PCA Dimensions", emb2d.shape[1])

# 📊 DATASET
elif page == "Dataset Preview":
    st.header("Dataset Preview")
    rows = st.slider("Rows:", 5, 50, 10)
    st.dataframe(df.head(rows), use_container_width=True)

# 📉 PCA
elif page == "PCA Visualization":
    st.header("PCA Visualization")

    n = st.slider("Points:", 500, min(10000, len(df)), 2000)

    fig = px.scatter(
        x=emb2d[:n, 0],
        y=emb2d[:n, 1],
        color=df["AGE_YRS"].head(n),
        opacity=0.6,
        color_continuous_scale="Viridis",
        labels={"x": "PCA-1", "y": "PCA-2"}
    )

    st.plotly_chart(fig, use_container_width=True)

# 🔍 SEMANTIC SEARCH
elif page == "Semantic Search":
    st.header("Semantic Search")

    query = st.text_input("Search symptoms")

    if query:
        k = st.slider("Top K", 1, 20, 5)

        with st.spinner("Searching..."):
            try:
                res = requests.get(
                    f"{API_URL}/search",
                    params={"query": query, "k": k}
                )
                data = res.json()

                for row in data["results"]:
                    with st.expander(f"VAERS {row['VAERS_ID']}"):
                        st.write(row["symptoms_normalized"])

            except:
                st.error("Backend not running 🚨")

# ⚖️ SEVERITY + 🤖 MISTRAL
elif page == "Severity Prediction":
    st.header("Severity Prediction")

    text = st.text_area("Enter symptoms")

    use_llm = st.checkbox("Explain with Mistral (Ollama)")

    if st.button("Predict"):
        if not text.strip():
            st.warning("Enter symptoms")
        else:
            with st.spinner("Predicting..."):
                try:
                    res = requests.get(
                        f"{API_URL}/predict",
                        params={"text": text}
                    )
                    data = res.json()

                    pred = data["prediction"]
                    prob = data["probability"]

                    st.success(pred)
                    st.write(f"Probability: {prob:.2f}")

                    # =========================
                    # 🔥 OLLAMA INTEGRATION
                    # =========================
                    if use_llm:
                        with st.spinner("Explaining with Mistral..."):
                            try:
                                import ollama

                                prompt = f"""
Explain this prediction in simple terms.

Symptoms: {text}
Prediction: {pred}
Probability: {prob:.2f}

Keep it short. No medical advice.
"""

                                response = ollama.chat(
                                    model="mistral",
                                    messages=[{"role": "user", "content": prompt}],
                                    options={"temperature": 0.2}
                                )

                                st.info(response["message"]["content"])

                            except Exception as e:
                                st.warning("Ollama not running or Mistral not installed ⚠️")
                                st.caption(str(e))

                except:
                    st.error("Backend not running 🚨")

