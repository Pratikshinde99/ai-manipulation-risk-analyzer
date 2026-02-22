"""
train.py
---------
End-to-end training pipeline for the
AI-Based Digital Content Manipulation Risk Detector.

Steps:
  1. Generate / load dataset
  2. Preprocess text  (lowercase, remove punctuation, stopwords, tokenise)
  3. Feature engineering:
       a. TF-IDF vectorizer (top 5000 terms, unigrams + bigrams)
       b. TextBlob sentiment polarity   [-1.0 … +1.0]
       c. TextBlob sentiment subjectivity [0.0 … 1.0]
     → Horizontally stack all features into one matrix
  4. Train Logistic Regression classifier (multiclass)
  5. Evaluate – accuracy, precision, recall, F1, confusion matrix
  6. Save model + vectorizer as .pkl files
"""

import os
import re
import string
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from textblob import TextBlob

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ── Download NLTK resources (only if missing) ─────────────────────────────────
print("📦 Checking NLTK resources …")
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
VADER = SentimentIntensityAnalyzer()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

DATASET_PATH    = os.path.join(DATA_DIR, "dataset.csv")
MODEL_PATH      = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
CM_IMG_PATH     = os.path.join(MODEL_DIR, "confusion_matrix.png")

# ── Step 1 – Load or generate dataset ────────────────────────────────────────
print("\n📂 Loading dataset …")
if not os.path.exists(DATASET_PATH):
    print("   Dataset not found – generating synthetic data …")
    from data.generate_dataset import build_dataset
    import csv
    dataset = build_dataset()
    with open(DATASET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(dataset)
    print(f"   Dataset generated → {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
print(f"   Rows loaded : {len(df)}")
print(f"   Class distribution:\n{df['label'].value_counts().to_string()}\n")

# ── Step 2 – Text preprocessing ──────────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    Clean a single text string:
      • Lowercase
      • Remove punctuation
      • Tokenise
      • Remove stopwords
    Returns a space-joined cleaned string.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


print("🔧 Preprocessing text …")
df["clean_text"] = df["text"].apply(preprocess)
print("   Preprocessing complete.\n")

# ── Step 3a – TF-IDF feature extraction ──────────────────────────────────────
print("📐 Vectorising with TF-IDF (max_features=5000, ngram 1-2) …")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(df["clean_text"])          # sparse: (N, 5000)
print(f"   TF-IDF matrix shape : {X_tfidf.shape}")

# ── Step 3b – TextBlob sentiment features ────────────────────────────────────
print("💬 Extracting sentiment features (VADER compound + TextBlob subjectivity) …")

def get_sentiment(text: str):
    """
    Returns (vader_compound, textblob_subjectivity) for the ORIGINAL (uncleaned) text.

    • VADER compound   : -1.0 (very negative) → +1.0 (very positive)
        Best for informal / social-media text — handles ALL CAPS, !, negations.
    • TextBlob subjectivity: 0.0 (objective) → 1.0 (subjective)
        Measures how opinion-based vs fact-based the text is.
    """
    vader_compound   = VADER.polarity_scores(text)["compound"]   # -1 … +1
    tb_subjectivity  = TextBlob(text).sentiment.subjectivity      #  0 … +1
    return vader_compound, tb_subjectivity

sentiments   = df["text"].apply(get_sentiment)
polarity     = np.array([s[0] for s in sentiments]).reshape(-1, 1)   # VADER compound
subjectivity = np.array([s[1] for s in sentiments]).reshape(-1, 1)   # TextBlob subjectivity

print(f"   VADER compound   — mean: {polarity.mean():.3f}  std: {polarity.std():.3f}")
print(f"   TB subjectivity  — mean: {subjectivity.mean():.3f}  std: {subjectivity.std():.3f}")

# Per-class averages — useful sanity check
df["vader_compound"]   = polarity
df["tb_subjectivity"]  = subjectivity
print("\n   Per-class sentiment averages:")
print(df.groupby("label")[["vader_compound", "tb_subjectivity"]].mean().round(3).to_string())
print()

# ── Step 3c – Combine TF-IDF + sentiment into one feature matrix ──────────────
# Convert the dense arrays to sparse so we can horizontally stack with X_tfidf
X_sentiment = sp.csr_matrix(np.hstack([polarity, subjectivity]))  # (N, 2)
X = sp.hstack([X_tfidf, X_sentiment])                             # (N, 5002)
y = df["label"]
print(f"   Combined feature matrix shape: {X.shape}  "
      f"(5000 TF-IDF + 2 sentiment)\n")

# ── Step 4 – Train / test split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"   Train set : {X_train.shape[0]} samples")
print(f"   Test  set : {X_test.shape[0]} samples\n")

# ── Step 5 – Model training ───────────────────────────────────────────────────
print("🤖 Training Logistic Regression …")
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    multi_class="multinomial",
    C=1.0,
    random_state=42,
)
model.fit(X_train, y_train)
print("   Training complete.\n")

# ── Step 6 – Evaluation ───────────────────────────────────────────────────────
print("📊 Evaluating model …")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n   Accuracy : {accuracy * 100:.2f}%\n")
print("   Classification Report:")
print(classification_report(y_test, y_pred, target_names=["High", "Low", "Moderate"]))

# Confusion matrix
labels = ["Low", "Moderate", "High"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax,
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix – Manipulation Risk Detector (TF-IDF + Sentiment)", fontsize=12)
plt.tight_layout()
plt.savefig(CM_IMG_PATH)
print(f"\n   Confusion matrix saved → {CM_IMG_PATH}")

# ── Step 7 – Save artefacts ───────────────────────────────────────────────────
print("\n💾 Saving model and vectorizer …")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"   Model saved      → {MODEL_PATH}")

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)
print(f"   Vectorizer saved → {VECTORIZER_PATH}")

print("\n✅ Training pipeline finished successfully!")
print("   Features used: TF-IDF (5000) + Sentiment Polarity + Sentiment Subjectivity")
