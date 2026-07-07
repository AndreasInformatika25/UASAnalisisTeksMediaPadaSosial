"""
Fintech Pinjol Insight Dashboard
=================================
Streamlit app for the SVM+TF-IDF classifier, LDA topic modeling, and
Social Network Analysis pipeline built in UAS_ATMSFinall.ipynb.

Design: pure light-mode "fintech" theme (white cards, green accents).
Data: read directly from local files in the project folder — no upload UI.
Modeling: all parameters (random_state, test sizes, vectorizer settings,
stopword lists) mirror the notebook exactly so results stay consistent.
"""

import re

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

# ──────────────────────────────────────────────────────────────────────────
# CONFIG / CONSTANTS
# ──────────────────────────────────────────────────────────────────────────

DATASET_PATH = "dataset.csv"
MODEL_PATH = "model_svm_pinjol.pkl"
VECTORIZER_PATH = "vectorizer_tfidf.pkl"
SNA_HTML_PATH = "sna_graph.html"
SNA_GRAPH_PATH = "sna_graph.gexf"

LDA_K_RANGE = range(2, 11)

# Stopwords used only for the side-by-side Korban vs Bot wordclouds
# (matches the notebook's "VISUALIZATION DATA" wordcloud section)
KATA_PASARAN_WORDCLOUD = [
    "tidak", "bayar", "gagal", "orang", "iya", "gitu", "gimana",
    "bukan", "pakai", "lain", "dan", "di", "yang", "buat", "ada",
    "ini", "itu", "udah", "dari", "kalau", "aja", "sama", "aku",
    "pinjol", "pinjaman", "online", "dana", "cepat", "cair",
]

# Keyword dictionary used for automatic topic labeling (notebook Bagian 8)
KAMUS_KATEGORI = {
    "Pelaporan Polisi & Judi Online": ["korban", "ilegal", "lapor", "judi", "daring", "polri", "kasih", "terima", "silah"],
    "Regulasi OJK & Ekonomi": ["ojk", "lain", "kredit", "utang", "grup", "ekonomi", "tau", "tingkat"],
    "Teror Debt Collector (DC)": ["debt", "collector", "lapang", "rumah", "teror", "yah", "tagih", "fc", "legal", "adakami"],
    "Data Pribadi & Aplikasi Pinjol": ["data", "hutang", "jangan", "aplikasi", "orang", "teman"],
    "Kendala Gaji & Pekerjaan": ["iya", "gimana", "hasil", "juta", "mutia", "oh", "kerja", "belum", "gaji", "tugas", "uang"],
    "Bot Promosi Jasa & Konsultasi": ["amp", "jasa", "zonauang", "ayo", "aman", "bantu", "konsultasi"],
    "Dampak Bunga & Mental (Bundir)": ["bukan", "hidup", "bunga", "bunuh", "cari"],
    "Kasus UKT Kampus & BI Checking": ["kakak", "gitu", "usaha", "bi", "coba", "atur", "ukt", "cek", "kampus"],
    "Promosi Pinjol Cepat Cair": ["dana", "butuh", "cepat", "jamin", "aju", "mudah", "cair", "kartu", "proses", "duduk", "tanda"]
}

# Optional: Sastrawi stemmer for the inference tab (Tab 4). See clean_text()
# docstring below for why this is an assumption, not a guaranteed match.
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _STEMMER = StemmerFactory().create_stemmer()
except ImportError:
    _STEMMER = None


# ──────────────────────────────────────────────────────────────────────────
# THEME / CSS
# ──────────────────────────────────────────────────────────────────────────

def inject_custom_css() -> None:
    """Force a pure light-mode fintech look: white cards, green accents,
    soft shadows. No dark-mode toggle is exposed anywhere in the app."""
    st.markdown(
        """
        <style>
        :root { color-scheme: light only; }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #FFFFFF !important;
            color: #1A1A1A !important;
        }
        [data-testid="stSidebar"] { background-color: #FFFFFF !important; }
        h1, h2, h3, h4, h5, h6, p, span, label { color: #1A1A1A; }

        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #F5F5F5;
            border-radius: 10px 10px 0 0;
            padding: 10px 18px;
            color: #1A1A1A;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2ECC71 !important;
            color: #FFFFFF !important;
        }

        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #EFEFEF;
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
        }
        div[data-testid="stMetricValue"] { color: #1A1A1A; }
        div[data-testid="stMetricLabel"] { color: #555555; }

        .stButton > button {
            background-color: #2ECC71;
            color: #FFFFFF;
            border: none;
            border-radius: 10px;
            padding: 0.55em 1.4em;
            font-weight: 600;
        }
        .stButton > button:hover { background-color: #27AE60; color: #FFFFFF; }

        .card {
            background-color: #FFFFFF;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.06);
            border: 1px solid #F0F0F0;
            margin-bottom: 16px;
        }
        hr { border-top: 1px solid #EFEFEF; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────
# DATA / MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Memuat dataset...")
def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the cleaned pinjol dataset from the local project folder.

    Mirrors the cleaning steps the notebook applies before feature
    extraction (Section 4): coerce label_kelas to numeric, drop rows that
    fail, drop any class with fewer than 2 samples (so the same
    train_test_split(..., stratify=y) call used later never breaks).
    """
    try:
        df = pd.read_csv(path, on_bad_lines="skip")
    except FileNotFoundError:
        st.error(f"❌ Dataset tidak ditemukan: `{path}`. Letakkan file ini di folder yang sama dengan app.py.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Gagal membaca dataset: {e}")
        st.stop()

    if "label_kelas" not in df.columns:
        st.error("❌ Kolom `label_kelas` tidak ditemukan di dataset.")
        st.stop()

    df["label_kelas"] = pd.to_numeric(df["label_kelas"], errors="coerce")
    df.dropna(subset=["label_kelas"], inplace=True)
    df["label_kelas"] = df["label_kelas"].astype(int)

    rare_classes = df["label_kelas"].value_counts()
    rare_classes = rare_classes[rare_classes < 2].index
    df = df[~df["label_kelas"].isin(rare_classes)]

    if "stemmed" in df.columns:
        df["stemmed"] = df["stemmed"].fillna("")

    return df.reset_index(drop=True)


@st.cache_resource(show_spinner="Memuat model SVM + TF-IDF...")
def load_svm_artifacts(model_path: str = MODEL_PATH, vectorizer_path: str = VECTORIZER_PATH):
    """Load the pre-trained SVM model and TF-IDF vectorizer (joblib .pkl).
    Returns (None, None) parts gracefully if a file is missing."""
    model, vectorizer = None, None

    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.warning(f"⚠️ File model `{model_path}` tidak ditemukan.")
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat model: {e}")

    try:
        vectorizer = joblib.load(vectorizer_path)
    except FileNotFoundError:
        st.warning(f"⚠️ File vectorizer `{vectorizer_path}` tidak ditemukan.")
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat vectorizer: {e}")

    return model, vectorizer


# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — DATASET & MODEL PERFORMANCE (SVM + TF-IDF)
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Mengevaluasi model SVM + TF-IDF...")
def evaluate_svm_model(df: pd.DataFrame, _model, _vectorizer):
    """Recreate the exact 80:20 split (random_state=42, stratify=y) from
    notebook Section 4, then evaluate the already-trained SVM model on the
    held-out test set so metrics match the Colab run exactly."""
    if _model is None or _vectorizer is None or "stemmed" not in df.columns:
        return None

    X = df["stemmed"].fillna("")
    y = df["label_kelas"]

    try:
        _, X_test_raw, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        st.error(f"Gagal membuat split data evaluasi: {e}")
        return None

    X_test_tfidf = _vectorizer.transform(X_test_raw)
    y_pred = _model.predict(X_test_tfidf)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["korban", "bot"], output_dict=True, zero_division=0
    )
    return {"metrics": metrics, "confusion_matrix": cm, "report": report}


def render_tab_dataset_model(df: pd.DataFrame, model, vectorizer) -> None:
    """Tab 1: metric cards, dataset explorer, SVM evaluation."""
    st.markdown("### 📊 Ringkasan Dataset")
    total_data = len(df)
    total_korban = int((df["label_kelas"] == 0).sum())
    total_bot = int((df["label_kelas"] == 1).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Data", f"{total_data:,}")
    c2.metric("Total Data Korban Pinjol", f"{total_korban:,}")
    c3.metric("Total Data Promo Bot Pinjol", f"{total_bot:,}")

    st.markdown("---")
    st.markdown("### 🔍 Dataset Explorer")
    all_columns = list(df.columns)
    default_cols = [c for c in ["text", "stemmed", "label_kelas"] if c in all_columns] or all_columns[:3]
    selected_cols = st.multiselect("Pilih kolom yang ingin ditampilkan:", all_columns, default=default_cols)
    if selected_cols:
        st.dataframe(df[selected_cols], use_container_width=True, height=340)
    else:
        st.info("Pilih minimal satu kolom untuk menampilkan tabel.")

    st.markdown("---")
    st.markdown("### 🤖 Evaluasi Model SVM + TF-IDF")

    if model is None or vectorizer is None:
        st.warning("Model atau vectorizer belum tersedia — letakkan `model_svm_pinjol.pkl` dan `vectorizer_tfidf.pkl` di folder project.")
        return

    result = evaluate_svm_model(df, model, vectorizer)
    if result is None:
        st.warning("Evaluasi tidak dapat dijalankan (cek kolom `stemmed`/`label_kelas`).")
        return

    m = result["metrics"]
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
    mc2.metric("Precision", f"{m['precision']*100:.2f}%")
    mc3.metric("Recall", f"{m['recall']*100:.2f}%")
    mc4.metric("F1-Score", f"{m['f1']*100:.2f}%")

    col_cm, col_report = st.columns(2)
    with col_cm:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        sns.heatmap(
            result["confusion_matrix"], annot=True, fmt="d", cmap="Greens",
            xticklabels=["korban", "bot"], yticklabels=["korban", "bot"], ax=ax,
        )
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig, use_container_width=True)

    with col_report:
        st.markdown("**Classification Report**")
        report_df = pd.DataFrame(result["report"]).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — TOPIC ANALYSIS & CLUSTERING (LDA)
# ──────────────────────────────────────────────────────────────────────────

def generate_wordcloud(text: str, colormap: str):
    """Build a WordCloud object for one label group's combined stemmed text."""
    if not text.strip():
        return None
    wc = WordCloud(
        width=800, height=500, background_color="#FFFFFF",
        stopwords=set(KATA_PASARAN_WORDCLOUD), colormap=colormap,
        max_words=100, random_state=42,
    )
    return wc.generate(text)


@st.cache_resource(show_spinner="Membangun representasi Bag-of-Words...")
def build_bow(df: pd.DataFrame):
    """CountVectorizer exactly as notebook Bagian 3 (used for LDA — not TF-IDF).
    No stopword filtering here on purpose, matching the original parameters."""
    cv = CountVectorizer(max_features=300, min_df=2, max_df=0.90, ngram_range=(1, 1))
    X = cv.fit_transform(df["stemmed"].fillna(""))
    terms = cv.get_feature_names_out()
    return cv, X, terms


@st.cache_data(show_spinner="Mencari jumlah topik optimal (elbow / perplexity)...")
def find_optimal_k(_X, k_range=LDA_K_RANGE):
    """Elbow/perplexity search exactly as notebook Bagian 4."""
    perplexities, log_likelihoods = [], []
    for k in k_range:
        lda_tmp = LatentDirichletAllocation(
            n_components=k, max_iter=15, learning_method="online", random_state=42
        )
        lda_tmp.fit(_X)
        perplexities.append(lda_tmp.perplexity(_X))
        log_likelihoods.append(lda_tmp.score(_X))
    best_k = list(k_range)[int(np.argmin(perplexities))]
    return list(k_range), perplexities, log_likelihoods, best_k


@st.cache_resource(show_spinner="Melatih model LDA final...")
def fit_final_lda(_X, n_topics: int):
    """Final LDA fit exactly as notebook Bagian 5."""
    lda = LatentDirichletAllocation(
        n_components=n_topics, max_iter=30, learning_method="online",
        doc_topic_prior=None, topic_word_prior=None, random_state=42,
    )
    lda.fit(_X)
    doc_topic_matrix = lda.transform(_X)
    return lda, doc_topic_matrix


def label_topics(lda, terms, n_topics: int) -> dict:
    """Auto-label each topic via keyword-overlap scoring (notebook Bagian 8)."""
    topic_labels = {}
    for k in range(n_topics):
        top_idx = lda.components_[k].argsort()[::-1][:15]
        top_words = [terms[i] for i in top_idx]
        label_terpilih, max_score = "Diskusi Umum / Lainnya", 0
        for kategori, kata_kunci in KAMUS_KATEGORI.items():
            score = sum(1 for w in top_words if w in kata_kunci)
            if score > max_score:
                max_score, label_terpilih = score, kategori
        topic_labels[k] = label_terpilih
    return topic_labels


def render_tab_lda(df: pd.DataFrame) -> None:
    """Tab 2: wordclouds, K-optimal chart, top 5 clusters, distribution charts."""
    st.markdown("### ☁️ Perbandingan Wordcloud: Korban vs Bot")
    col1, col2 = st.columns(2)
    teks_korban = " ".join(df[df["label_kelas"] == 0]["stemmed"].dropna().astype(str)) if "stemmed" in df.columns else ""
    teks_bot = " ".join(df[df["label_kelas"] == 1]["stemmed"].dropna().astype(str)) if "stemmed" in df.columns else ""

    with col1:
        st.markdown("**Korban Pinjol**")
        wc_k = generate_wordcloud(teks_korban, colormap="Greens")
        if wc_k is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(wc_k, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Tidak ada teks korban yang tersedia.")

    with col2:
        st.markdown("**Bot Promo Pinjol**")
        wc_b = generate_wordcloud(teks_bot, colormap="YlGn")
        if wc_b is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(wc_b, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Tidak ada teks bot yang tersedia.")

    if "stemmed" not in df.columns:
        st.error("Kolom `stemmed` tidak ditemukan — analisis LDA tidak dapat dijalankan.")
        return

    st.markdown("---")
    st.markdown("### 📈 Evaluasi Jumlah Topik Optimal (Elbow / Perplexity)")
    cv, X_bow, terms = build_bow(df)
    k_values, perplexities, log_likelihoods, best_k = find_optimal_k(X_bow)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(k_values, perplexities, "o-", color="#2ECC71", lw=2.2)
    axes[0].axvline(x=best_k, color="#1A1A1A", linestyle="--", alpha=0.6, label=f"K optimal = {best_k}")
    axes[0].set_title("Perplexity vs K (↓ lebih baik)")
    axes[0].set_xlabel("Jumlah Topik (K)")
    axes[0].legend()
    axes[1].plot(k_values, log_likelihoods, "s-", color="#0891B2", lw=2.2)
    axes[1].axvline(x=best_k, color="#1A1A1A", linestyle="--", alpha=0.6, label=f"K optimal = {best_k}")
    axes[1].set_title("Log-Likelihood vs K (↑ lebih baik)")
    axes[1].set_xlabel("Jumlah Topik (K)")
    axes[1].legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.success(f"Jumlah topik optimal berdasarkan perplexity terendah: **K = {best_k}**")

    n_topics = best_k
    lda, doc_topic_matrix = fit_final_lda(X_bow, n_topics)

    df = df.copy()
    df["dominant_topic"] = doc_topic_matrix.argmax(axis=1)
    df["topic_prob"] = doc_topic_matrix.max(axis=1)
    topic_labels = label_topics(lda, terms, n_topics)
    df["topic_label"] = df["dominant_topic"].map(topic_labels)

    st.markdown("---")
    st.markdown("### 🏆 Top 5 Klaster Topik")
    topic_counts = df["dominant_topic"].value_counts()
    top5_topics = topic_counts.sort_values(ascending=False).head(5).index.tolist()

    for k in top5_topics:
        top_idx = lda.components_[k].argsort()[::-1][:10]
        top_words = [terms[i] for i in top_idx]
        st.markdown(
            f"""<div class="card">
            <b>Topik {k}: {topic_labels.get(k, '-')}</b><br>
            Jumlah dokumen: {int(topic_counts.get(k, 0))}<br>
            Top Words: {", ".join(top_words)}
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🔠 Distribusi Kata per Topik")
    topic_choice = st.selectbox(
        "Pilih topik:", options=list(range(n_topics)),
        format_func=lambda k: f"Topik {k}: {topic_labels.get(k, '-')}",
    )
    top_idx = lda.components_[topic_choice].argsort()[::-1][:12]
    top_words = [terms[i] for i in top_idx]
    top_vals = lda.components_[topic_choice][top_idx]
    top_vals = top_vals / top_vals.sum()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(top_words)), top_vals[::-1], color="#2ECC71", edgecolor="white")
    ax.set_yticks(range(len(top_words)))
    ax.set_yticklabels(top_words[::-1])
    ax.set_xlabel("Bobot (normalized)")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🗺️ Visualisasi Pendukung")
    v1, v2, v3 = st.columns(3)

    with v1:
        st.markdown("**Heatmap Topic-Word**")
        top_global_idx = lda.components_.sum(axis=0).argsort()[::-1][:15]
        top_global_words = [terms[i] for i in top_global_idx]
        heatmap_data = lda.components_[:, top_global_idx]
        heatmap_norm = heatmap_data / heatmap_data.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            heatmap_norm, cmap="Greens", xticklabels=top_global_words,
            yticklabels=[f"T{k}" for k in range(n_topics)], ax=ax,
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with v2:
        st.markdown("**Proporsi Topik**")
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        sorted_counts = topic_counts.sort_index()
        ax.pie(
            sorted_counts.values, labels=[f"T{k}" for k in sorted_counts.index],
            autopct="%1.0f%%", colors=sns.color_palette("Greens", n_topics),
        )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with v3:
        st.markdown("**Rata-rata Probabilitas Topik**")
        avg_prob = doc_topic_matrix.mean(axis=0)
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.bar([f"T{k}" for k in range(n_topics)], avg_prob, color="#2ECC71")
        ax.set_ylabel("Avg. Probability")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# TAB 3 — SOCIAL NETWORK ANALYSIS (SNA)
# ──────────────────────────────────────────────────────────────────────────

def load_sna_html(path: str = SNA_HTML_PATH):
    """Read the pre-generated PyVis HTML network file as raw text."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Gagal membaca file HTML SNA: {e}")
        return None


@st.cache_data(show_spinner="Menghitung metrik centrality...")
def compute_centrality_metrics(path: str = SNA_GRAPH_PATH):
    """Load the pre-exported GEXF graph and compute in-degree, out-degree,
    betweenness, and eigenvector centrality — the same metrics referenced
    in the notebook's SNA section."""
    try:
        G = nx.read_gexf(path)
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Gagal membaca file graph SNA: {e}")
        return None

    in_degree = nx.in_degree_centrality(G)
    out_degree = nx.out_degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {}

    def top_n(d, n=5):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]

    return {
        "in_degree": top_n(in_degree),
        "out_degree": top_n(out_degree),
        "betweenness": top_n(betweenness),
        "eigenvector": top_n(eigenvector),
    }


def render_tab_sna() -> None:
    """Tab 3: interactive PyVis HTML graph + centrality summary panels."""
    st.markdown("### 🕸️ Visualisasi Jaringan (Social Network Analysis)")
    html_content = load_sna_html()
    if html_content:
        components.html(html_content, height=600, scrolling=True)
    else:
        st.warning(f"⚠️ File `{SNA_HTML_PATH}` tidak ditemukan. Letakkan hasil export PyVis di folder project.")

    st.markdown("---")
    st.markdown("### 👑 Akun Paling Berpengaruh")
    metrics = compute_centrality_metrics()
    if metrics is None:
        st.warning(f"⚠️ File graph `{SNA_GRAPH_PATH}` tidak ditemukan — metrik centrality tidak dapat dihitung.")
        return

    mc1, mc2, mc3, mc4 = st.columns(4)
    panels = [
        (mc1, "Top In-Degree", metrics["in_degree"], "Akun paling banyak menerima mention"),
        (mc2, "Top Out-Degree", metrics["out_degree"], "Akun paling aktif menyebar mention"),
        (mc3, "Top Betweenness", metrics["betweenness"], "Akun sebagai jembatan informasi"),
        (mc4, "Top Eigenvector", metrics["eigenvector"], "Akun terhubung dengan node penting"),
    ]
    for col, title, items, caption in panels:
        with col:
            st.markdown(f"**{title}**")
            st.caption(caption)
            if items:
                for name, score in items:
                    st.write(f"@{name} — {score:.4f}")
            else:
                st.write("Tidak ada data.")


# ──────────────────────────────────────────────────────────────────────────
# TAB 4 — INDEPENDENT DETECTION (INFERENCE)
# ──────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Preprocess raw input text before TF-IDF transform + SVM prediction.

    ASSUMPTION: the notebook calls a `clean_text()` helper before this
    inference step, but its exact body wasn't included in the excerpt
    integrated here. This implementation (lowercase → strip
    URLs/mentions/hashtags/punctuation → optional Sastrawi stem) follows the
    conventions implied by the 'stemmed' column elsewhere in the notebook.
    If your real preprocessing differs, replace this function so Tab 4
    matches your model's actual training-time pipeline.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if _STEMMER is not None:
        text = _STEMMER.stem(text)
    return text


def render_tab_inference(model, vectorizer) -> None:
    """Tab 4: free-text input → preprocessing → SVM prediction → result card."""
    st.markdown("### 🔎 Deteksi Teks Baru")
    st.caption("Masukkan teks atau tweet terkait pinjol untuk dideteksi sebagai Curhatan Korban atau Promo Bot.")

    if model is None or vectorizer is None:
        st.warning("Model atau vectorizer belum tersedia — deteksi tidak dapat dijalankan.")
        return

    user_text = st.text_area(
        "Teks input:", height=140,
        placeholder="Contoh: Tolong jangan pakai aplikasi pinjol ini, DC-nya teror terus...",
    )

    if st.button("🚀 Deteksi Sekarang"):
        if not user_text.strip():
            st.error("Mohon masukkan teks terlebih dahulu.")
            return
        try:
            cleaned = clean_text(user_text)
            vectorized = vectorizer.transform([cleaned])
            pred = int(model.predict(vectorized)[0])
            proba = model.predict_proba(vectorized)[0] if hasattr(model, "predict_proba") else None

            reverse_mapping = {0: "Curhatan Korban Pinjol", 1: "Promo Bot Pinjol"}
            label_text = reverse_mapping.get(pred, "Tidak diketahui")

            border_color = "#2ECC71" if pred == 0 else "#E74C3C"
            icon = "✅" if pred == 0 else "🚨"
            st.markdown(
                f"""<div class="card" style="border-left:6px solid {border_color};">
                <h4 style="color:{border_color};margin:0;">{icon} {label_text}</h4>
                </div>""",
                unsafe_allow_html=True,
            )

            if proba is not None:
                st.markdown("**Probabilitas:**")
                pc1, pc2 = st.columns(2)
                pc1.metric("Korban", f"{proba[0]*100:.2f}%")
                pc2.metric("Bot", f"{proba[1]*100:.2f}%")
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Pinjol Insight Dashboard", page_icon="💳", layout="wide")
    inject_custom_css()

    st.markdown(
        """<div style="padding:6px 0 18px 0;">
        <h1 style="margin-bottom:0;">💳 Pinjol Insight Dashboard</h1>
        <p style="color:#555;">Deteksi Korban vs Bot Promosi Pinjaman Online — SVM + TF-IDF,
        LDA Topic Modeling, dan Social Network Analysis</p>
        </div>""",
        unsafe_allow_html=True,
    )

    df = load_dataset()
    model, vectorizer = load_svm_artifacts()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dataset & Model Performance",
        "🧠 Topic Analysis (LDA)",
        "🕸️ Social Network Analysis",
        "🔎 Independent Detection",
    ])

    with tab1:
        render_tab_dataset_model(df, model, vectorizer)
    with tab2:
        render_tab_lda(df)
    with tab3:
        render_tab_sna()
    with tab4:
        render_tab_inference(model, vectorizer)


if __name__ == "__main__":
    main()