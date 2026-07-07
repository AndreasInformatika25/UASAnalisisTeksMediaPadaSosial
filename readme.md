# 💳 Pinjol (Pinjaman Online) Insight Dashboard: Victim vs. Promotional Bot Detection

An interactive, web-based analytical dashboard designed to systematically detect and analyze social media discourse surrounding Online Loans (_PinJol_). This project integrates Natural Language Processing (NLP), Machine Learning (Support Vector Machine with TF-IDF), Topic Modeling (Latent Dirichlet Allocation), and Social Network Analysis (SNA) to empirically distinguish genuine victims' grievances from promotional bot spam.

## 🚀 Live Demo

The application is currently deployed and accessible to the public. You can experience the live environment via the following link:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sentimen-pinjamanonline-atms.streamlit.app/)

👉 **[Access the Pinjol Insight Dashboard](https://sentimen-pinjamanonline-atms.streamlit.app/)**

## ✨ Core Features

This dashboard offers four primary analytical modules:

1. **📊 Dataset & Model Performance**
   Presents a comprehensive overview of the empirical dataset (comprising 2,518 total records: 2,134 Victims and 384 Bots). This section features a Dataset Explorer and visualizes the evaluation metrics for the SVM + TF-IDF model, including the Confusion Matrix and Classification Report, achieving an exceptional **accuracy rate of 99.60%**.
2. **🧠 Topic Analysis (LDA)**
   Extracts and visualizes the primary conversational themes utilizing the Latent Dirichlet Allocation (LDA) algorithm. This module includes comparative Wordclouds (Victim vs. Bot), evaluation charts for determining the optimal number of topics (Elbow Method/Perplexity), and the distribution weights of words per topic cluster.
3. **🕸️ Social Network Analysis (SNA)**
   Maps the interaction networks among social media accounts using NetworkX and Pyvis. It displays interactive network graph visualizations and identifies the most influential nodes (accounts) based on key centrality metrics: _Top In-Degree, Out-Degree, Betweenness_, and _Eigenvector Centrality_.
4. **🔎 Independent Detection**
   A real-time inference testing module. Users can input new text or tweets to instantly determine whether the content is classified as a "Victim's Grievance" or a "Promotional Bot," accompanied by its calculated probability scores.

## 🛠️ Technology Stack & Dependencies

This project is built upon a robust ecosystem of modern Python libraries:

- **Web Framework:** `streamlit`, `streamlit.components.v1`
- **Machine Learning & NLP:**
  - `scikit-learn` (SVM, TF-IDF, CountVectorizer, LDA, Evaluation Metrics)
  - `nltk`, `Sastrawi` (Text Preprocessing & Indonesian Stemming)
  - `gensim` (Word2Vec)
  - `transformers`, `torch` (Advanced NLP / Transformer-based experiments)
- **Social Network Analysis:** `networkx`, `pyvis`
- **Data Visualization:** `matplotlib`, `seaborn`, `wordcloud`
- **Data Manipulation & Utilities:** `pandas`, `numpy`, `re`, `joblib`
