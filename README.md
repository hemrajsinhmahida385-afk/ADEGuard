#ADEGuard
# live working App
https://adeguard-nvvthuefjkvbvgjdru2yh6.streamlit.app/
# 📌 Problem Statement

Monitoring adverse drug events (ADEs) is a critical task in public health and pharmacovigilance.
The VAERS (Vaccine Adverse Event Reporting System) dataset contains 1.2M+ reports from the last 5 years. While this data is rich, its scale introduces key challenges:

🔍 Difficulty in searching and comparing reports efficiently

📊 Complexity in identifying patterns within high-dimensional data

⚖️ Limited ability to classify event severity in a transparent, explainable way

Mini-ADEGuard addresses these challenges by transforming raw VAERS data into an interactive, searchable, and explainable web application.

## 🎯 What the Project Does

🔍 Semantic Search → retrieve similar adverse events using SBERT embeddings

📊 Dataset Exploration → preview, filter, and interact with VAERS data

🧮 Severity Modelling → baseline classification with Logistic Regression

📉 Visualization → reduce embedding dimensions with PCA for pattern discovery

🌐 Deployment → accessible via Streamlit Cloud, no setup required

⚡ For reproducibility on GitHub, this repository includes a 50k-row sample of the VAERS dataset with all models trained on it.
The original project was built on the complete 1.2M+ records.

### ⚙️ Workflow
🗂️ Data Preparation

Cleaned VAERS dataset

Sampled 50k rows for training and deployment

🛠️ Feature Engineering

Generated text embeddings with SBERT MiniLM (384 dimensions)

Built sparse TF-IDF vectors for classification tasks

🤖 Model Training

Trained a Logistic Regression classifier to predict ADE severity

Stored embeddings for fast semantic similarity search

💻 App Development

Designed a Streamlit app with three main modules:

📊 Dataset Preview

📉 PCA Visualization

🔍 Semantic Search

🚀 Deployment

Hosted on Streamlit Cloud for interactive, real-time access

### 📸 Screenshots

Home Dashboard
![8abed9db-f929-4bc0-9a16-5f1d1b878b30](https://github.com/user-attachments/assets/11281a19-f3fa-4ae1-8e27-5ee81ee11b25)

Dataset Preview
![f2d4ebad-4e71-484a-9e7c-4222962c89c2](https://github.com/user-attachments/assets/be176052-f791-4a8a-9167-f2982467c6d4)

PCA Visualization
![e3da2940-8991-4cba-aada-6eb3a7631ae3](https://github.com/user-attachments/assets/2f427978-85b7-45d4-9068-c3b3c5742b3c)

Semantic Search
![b5106a93-f029-4725-a5e6-de13534a649c](https://github.com/user-attachments/assets/142f37cf-ccaf-441a-a621-b7450f20352a)

# 🚀 Run on Your Machine
# Clone repo
git clone https://github.com/<your-username>/ADEGuard.git
cd ADEGuard/notebook

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run mini-adeguard.py


Open your browser at http://localhost:8501

🌟 Highlights

📂 Processed and analyzed large-scale medical data (1.2M+ VAERS reports)

🔗 Designed an end-to-end pipeline: raw data → embeddings → model → visualization → deployment

🤖 Leveraged SBERT embeddings for semantic search, similarity matching, and clustering

🌐 Developed a production-ready Streamlit application, deployed and accessible via the web

🛠️ Demonstrated expertise in data engineering, NLP, machine learning, and full-stack deployment

👤 Author
