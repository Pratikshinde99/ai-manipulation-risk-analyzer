"""
app.py
-------
Streamlit web application for the
AI-Based Digital Content Manipulation Risk Detector.

Features:
  • Paste any text (social-media post, headline, news snippet)
  • Click "Analyse" to get:
      - Predicted risk category (Low / Moderate / High)
      - Confidence score (%)
      - Colour-coded result card (Green / Yellow / Red)
  • Sidebar with project info and example texts
"""

import os
import re
import string
import pickle
import warnings

import numpy as np
import scipy.sparse as sp
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

warnings.filterwarnings("ignore")

# ── NLTK setup ────────────────────────────────────────────────────────────────
for resource in ["stopwords", "punkt", "punkt_tab", "vader_lexicon"]:
    try:
        nltk.data.find(
            f"tokenizers/{resource}" if resource.startswith("punkt")
            else f"sentiment/{resource}" if resource == "vader_lexicon"
            else f"corpora/{resource}"
        )
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
VADER      = SentimentIntensityAnalyzer()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "model", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Content Manipulation Risk Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font ─────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background ──────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Main title ──────────────────────────── */
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* ── Text area ───────────────────────────── */
    .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        font-size: 0.95rem !important;
    }

    /* ── Analyse button ──────────────────────── */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2.2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        white-space: nowrap !important;
        min-width: 160px !important;
        width: fit-content !important;
        height: auto !important;
        line-height: 1.5 !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(124,58,237,0.5) !important;
    }

    /* ── Result cards ────────────────────────── */
    .result-card {
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-top: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 40px rgba(0,0,0,0.35);
        animation: fadeIn 0.4s ease;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .card-low      { background: linear-gradient(135deg, #065f46, #059669); border: 1px solid #10b981; }
    .card-moderate { background: linear-gradient(135deg, #78350f, #d97706); border: 1px solid #f59e0b; }
    .card-high     { background: linear-gradient(135deg, #7f1d1d, #dc2626); border: 1px solid #ef4444; }

    .card-label {
        font-size: 1.1rem;
        font-weight: 500;
        color: rgba(255,255,255,0.8);
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .card-risk {
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }
    .card-confidence {
        font-size: 1.25rem;
        color: rgba(255,255,255,0.85);
        font-weight: 500;
    }
    .card-icon {
        font-size: 3.5rem;
        margin-bottom: 0.75rem;
    }
    .card-description {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.7);
        margin-top: 0.8rem;
        line-height: 1.5;
    }

    /* ── Meter bar ───────────────────────────── */
    .meter-container {
        margin-top: 1.5rem;
    }
    .meter-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0.4rem;
        font-weight: 500;
    }
    .meter-bar-bg {
        background: rgba(255,255,255,0.1);
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
    }
    .meter-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.7s ease;
    }

    /* ── Info boxes ──────────────────────────── */
    .info-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    /* ── Sidebar ─────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(15,12,41,0.85) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }

    /* ── Divider ─────────────────────────────── */
    hr {
        border-color: rgba(255,255,255,0.1) !important;
        margin: 1.5rem 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model artefacts ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artefacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artefacts()

# ── Text preprocessing (must match train.py) ──────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 About This Tool")
    st.markdown(
        """
        <div class="info-box">
        This tool uses a <b>Logistic Regression</b> classifier trained on 200
        synthetic text samples to detect the manipulation risk level in
        digital content such as social-media posts, news headlines, and articles.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎨 Risk Levels")
    st.markdown(
        """
        <div class="info-box">
        🟢 <b>Low Risk</b> — Neutral, factual content<br><br>
        🟡 <b>Moderate Risk</b> — Emotional or persuasive language<br><br>
        🔴 <b>High Risk</b> — Fear-based, urgency-driven, or sensational content
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 Example Texts")
    examples = {
        "📰 Low (factual)": "The city council met yesterday to discuss road repair budgets.",
        "⚠️ Moderate (persuasive)": "You won't believe what this politician said to enrage millions!",
        "🚨 High (sensational)": "URGENT: The government is secretly poisoning water — act NOW before it's too late!",
    }
    selected_example = st.selectbox("Try an example →", ["— select —"] + list(examples.keys()))

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.78rem;color:#475569;text-align:center;'>Built with Streamlit · Scikit-learn</div>",
        unsafe_allow_html=True,
    )

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🔍 AI Content Manipulation Risk Detector</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Paste any text below and let AI assess its manipulation risk level.</p>',
    unsafe_allow_html=True,
)

if model is None or vectorizer is None:
    st.error(
        "⚠️ No trained model found. Please run `python train.py` first to train and save the model.",
        icon="🚫",
    )
    st.stop()

# ── Text input ────────────────────────────────────────────────────────────────
default_text = ""
if selected_example and selected_example != "— select —":
    default_text = examples[selected_example]

col_main, col_pad = st.columns([3, 1])
with col_main:
    user_input = st.text_area(
        label="📝 Input Text",
        value=default_text,
        height=180,
        placeholder="Paste a social-media post, news headline, or any text here …",
        label_visibility="visible",
    )

    analyse_btn = st.button("🔍 Analyse", use_container_width=False)

# ── Prediction ────────────────────────────────────────────────────────────────
RISK_CONFIG = {
    "Low": {
        "css_class":   "card-low",
        "icon":        "🟢",
        "color":       "#10b981",
        "description": "This content appears neutral and factual with no significant manipulation indicators.",
    },
    "Moderate": {
        "css_class":   "card-moderate",
        "icon":        "🟡",
        "color":       "#f59e0b",
        "description": "This content contains emotional or persuasive language that may influence reader perception.",
    },
    "High": {
        "css_class":   "card-high",
        "icon":        "🔴",
        "color":       "#ef4444",
        "description": "This content uses fear, urgency, or sensationalism — high risk of manipulation.",
    },
}

if analyse_btn:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before clicking Analyse.", icon="✍️")
    else:
        with st.spinner("Analysing …"):
            # ── TF-IDF features ───────────────────────────────────
            clean    = preprocess(user_input)
            X_tfidf  = vectorizer.transform([clean])          # sparse (1, 5000)

            # ── TextBlob sentiment features ───────────────────────
            # VADER compound: much better than TextBlob for informal/social text
            vader_scores  = VADER.polarity_scores(user_input)
            polarity      = vader_scores["compound"]             # -1.0 … +1.0
            subjectivity  = TextBlob(user_input).sentiment.subjectivity  # 0 … 1
            X_sentiment  = sp.csr_matrix([[polarity, subjectivity]])  # (1, 2)

            # ── Combined feature vector (must match train.py) ─────
            X_combined = sp.hstack([X_tfidf, X_sentiment])   # (1, 5002)

            probs      = model.predict_proba(X_combined)[0]
            classes    = model.classes_
            pred_label = classes[np.argmax(probs)]
            confidence = float(np.max(probs)) * 100

        cfg = RISK_CONFIG[pred_label]

        # ── Result card ───────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="result-card {cfg['css_class']}">
                <div class="card-icon">{cfg['icon']}</div>
                <div class="card-label">Manipulation Risk Level</div>
                <div class="card-risk">{pred_label} Risk</div>
                <div class="card-confidence">Confidence: {confidence:.1f}%</div>
                <div class="card-description">{cfg['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Per-class probability bars ────────────────────────────────────────
        st.markdown("<div class='meter-container'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#94a3b8;font-size:0.9rem;margin-top:1.5rem;font-weight:500;'>"
            "📊 Class Probabilities</p>",
            unsafe_allow_html=True,
        )
        bar_colors = {"Low": "#10b981", "Moderate": "#f59e0b", "High": "#ef4444"}
        for cls, prob in zip(classes, probs):
            pct = prob * 100
            st.markdown(
                f"""
                <div style="margin-bottom:0.9rem;">
                    <div class="meter-label">{cls} Risk — {pct:.1f}%</div>
                    <div class="meter-bar-bg">
                        <div class="meter-bar-fill"
                             style="width:{pct}%;background:{bar_colors.get(cls,'#6366f1')};">
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Analysed text preview ─────────────────────────────────────────────
        with st.expander("🔎 View analysed text details"):
            st.markdown(f"**Original (first 400 chars):** {user_input[:400]}{'…' if len(user_input)>400 else ''}")
            st.markdown(f"**After preprocessing:** {clean[:400]}{'…' if len(clean)>400 else ''}")
            st.markdown(f"**Tokens used:** {len(clean.split())}")

            # Sentiment detail
            st.markdown("---")
            st.markdown("**📈 Sentiment Scores  *(VADER compound + TextBlob subjectivity)*")
            # VADER compound interpretation
            if polarity >= 0.05:
                polarity_label = "Positive 😊"
            elif polarity <= -0.05:
                polarity_label = "Negative 😠"
            else:
                polarity_label = "Neutral 😐"
            subjectivity_label = "Subjective 💭" if subjectivity > 0.5 else "Objective 📋"
            col_p, col_s = st.columns(2)
            col_p.metric(
                label="VADER Compound (Polarity)",
                value=f"{polarity:+.3f}",
                help="VADER compound score: −1.0 = very negative · 0 = neutral · +1.0 = very positive. "
                     "Handles ALL CAPS, !, negations and social-media language.",
            )
            col_p.caption(polarity_label)
            col_s.metric(
                label="Subjectivity",
                value=f"{subjectivity:.3f}",
                help="TextBlob subjectivity: 0.0 = fully objective/factual · 1.0 = highly subjective/opinionated",
            )
            col_s.caption(subjectivity_label)
