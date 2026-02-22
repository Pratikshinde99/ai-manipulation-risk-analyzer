<div align="center">

# 🔍 AI-Based Digital Content Manipulation Risk Detector

### Detect misinformation, propaganda, and sensationalism in text — instantly.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-154360?style=for-the-badge)](https://nltk.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> A production-ready NLP system that analyses digital text — social media posts, news headlines, and articles — and classifies the **manipulation risk level** into **Low**, **Moderate**, or **High**, with a real-time confidence score.

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Risk Categories](#-risk-categories)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Deployment — Streamlit Cloud](#-deployment--streamlit-cloud)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

Digital content manipulation — including **misinformation**, **propaganda**, **fear-baiting**, and **clickbait** — is one of the most pressing challenges in the age of social media. This project provides an end-to-end machine learning pipeline that:

1. **Ingests** any piece of text (tweet, headline, news snippet, blog post)
2. **Preprocesses** it using NLP best practices
3. **Extracts features** using TF-IDF + VADER sentiment + TextBlob subjectivity
4. **Classifies** the manipulation risk with a calibrated probability score
5. **Presents** results through a polished, colour-coded web interface

Built as part of the **Microsoft Elevate AI Internship Programme**, this project demonstrates real-world application of classical NLP, feature engineering, and ML model deployment.

---

## 🌐 Live Demo

> Deploy your own instance in minutes — see [Streamlit Cloud Deployment](#-deployment--streamlit-cloud) below.

**Run locally:**
```bash
py -m streamlit run app.py
```
Then open → **http://localhost:8501**

---

## 🚦 Risk Categories

| Level | Label | Description | Example |
|:-----:|-------|-------------|---------|
| 🟢 | **Low Risk** | Neutral, factual reporting with no manipulation indicators | *"The city council approved road maintenance funding for Q2."* |
| 🟡 | **Moderate Risk** | Emotional or persuasive language that may bias the reader | *"You won't believe what this politician said to enrage millions!"* |
| 🔴 | **High Risk** | Fear-based, urgency-driven, or sensational content designed to manipulate | *"URGENT: Government secretly poisoning water — act NOW before it's too late!"* |

---

## ✨ Features

### 🔬 ML Pipeline
- **Synthetic dataset** — 200 carefully crafted, balanced samples across all 3 risk classes
- **Text preprocessing** — Lowercasing → punctuation removal → digit stripping → NLTK tokenisation → stopword removal
- **Hybrid feature engineering:**
  - **TF-IDF Vectorizer** — top 5,000 terms, unigrams + bigrams
  - **VADER Compound Score** — purpose-built for social-media text; handles ALL CAPS, `!`, negations, slang
  - **TextBlob Subjectivity** — measures how opinion-based vs. fact-based the content is
- **Logistic Regression** — multinomial, `lbfgs` solver, `C=1.0`, stratified 80/20 split
- **Evaluation** — accuracy, precision, recall, F1-score, and a confusion matrix heatmap

### 🖥️ Web Application
- Dark-mode UI with glassmorphism aesthetics and gradient animations
- **Colour-coded result cards** — Green / Yellow / Red with animated fade-in
- **Per-class probability bars** for full prediction transparency
- **Sentiment analysis panel** — VADER polarity + TextBlob subjectivity displayed with labelled metric cards
- **One-click examples** via sidebar dropdown
- **Preprocessed text expander** — see exactly what the model sees

---

## ⚙️ How It Works

```
Input Text
    │
    ▼
┌─────────────────────────────────────────────┐
│           TEXT PREPROCESSING                │
│  lowercase → strip punctuation → tokenise   │
│        → remove stopwords (NLTK)            │
└───────────────────┬─────────────────────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
   ┌─────────────┐    ┌──────────────────────┐
   │  TF-IDF     │    │  Sentiment Features  │
   │ Vectorizer  │    │  ─────────────────── │
   │ (5000 terms │    │  VADER compound score│
   │  1-2 ngram) │    │  TextBlob subjectivity│
   └──────┬──────┘    └──────────┬───────────┘
          │                      │
          └──────────┬───────────┘
                     ▼
          ┌────────────────────┐
          │  scipy.sparse      │
          │  hstack → (N,2302) │
          └────────┬───────────┘
                   ▼
        ┌─────────────────────┐
        │  Logistic Regression│
        │  (multinomial)      │
        └────────┬────────────┘
                 ▼
     ┌───────────────────────┐
     │  Risk Label +         │
     │  Confidence Score (%) │
     └───────────────────────┘
```

---

## 📁 Project Structure

```
AI-Based Digital Content Manipulation Risk Detector/
│
├── 📄 app.py                     ← Streamlit web application (UI + real-time inference)
├── 📄 train.py                   ← End-to-end training pipeline
├── 📄 requirements.txt           ← All Python dependencies (Streamlit Cloud ready)
├── 📄 README.md                  ← You are here
│
├── 📂 data/
│   ├── generate_dataset.py       ← Synthetic dataset generator (200 samples)
│   ├── dataset.csv               ← Generated dataset (auto-created on first run)
│   └── __init__.py
│
└── 📂 model/
    ├── model.pkl                 ← Trained Logistic Regression model
    ├── vectorizer.pkl            ← Fitted TF-IDF vectorizer
    └── confusion_matrix.png      ← Evaluation heatmap (auto-generated)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Web Framework | Streamlit |
| ML / Feature Engineering | scikit-learn (TF-IDF + Logistic Regression) |
| NLP — Preprocessing | NLTK (tokenisation, stopwords) |
| NLP — Sentiment (Polarity) | NLTK VADER `SentimentIntensityAnalyzer` |
| NLP — Sentiment (Subjectivity) | TextBlob |
| Data Handling | pandas, NumPy, SciPy (sparse matrices) |
| Visualisation | Matplotlib, Seaborn |
| Model Persistence | Python `pickle` |

---

## ⚡ Installation

### Prerequisites
- Python **3.9 or higher**
- `pip` / `py` available in your terminal

### 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-content-manipulation-detector.git
cd ai-content-manipulation-detector
```

### 2 — Create a virtual environment *(recommended)*
```bash
# Windows
py -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Download NLP corpora
These are downloaded automatically on first run, but you can pre-fetch them:
```bash
py -m textblob.download_corpora
```

---

## 🚀 Usage

### Step 1 — Train the model
```bash
py train.py
```

**What this does:**
- Auto-generates `data/dataset.csv` (200 synthetic samples) if it doesn't exist
- Preprocesses all text samples
- Fits TF-IDF vectorizer + extracts VADER & TextBlob sentiment features
- Trains a Logistic Regression classifier (80 / 20 stratified split)
- Prints accuracy, precision, recall, and F1-score
- Saves `model/model.pkl` and `model/vectorizer.pkl`
- Saves `model/confusion_matrix.png`

### Step 2 — Launch the web app
```bash
py -m streamlit run app.py
```

Open your browser at **http://localhost:8501**

> **Important:** Always retrain (`py train.py`) before restarting the app whenever you change the feature pipeline. The Streamlit app caches the model in memory — restart the server after retraining to load the updated model.

---

## 📊 Model Performance

> Results on the held-out 20% test set (40 samples, stratified):

```
Accuracy: ~97%

              precision    recall  f1-score   support

        High       0.93      1.00      0.96        13
         Low       1.00      1.00      1.00        14
    Moderate       1.00      0.93      0.96        13

    accuracy                           0.97        40
   macro avg       0.98      0.98      0.97        40
weighted avg       0.98      0.97      0.97        40
```

### Why VADER over TextBlob for polarity?

| Text | TextBlob Polarity | VADER Compound |
|------|:-----------------:|:--------------:|
| `"You won't believe what this politician said to enrage millions!"` | `+0.000` ❌ | `−0.40` ✅ |
| `"URGENT! Government secretly poisoning water — act NOW!"` | `+0.000` ❌ | `−0.65` ✅ |
| `"Scientists publish new findings on coral reef ecosystems."` | `0.000` | `+0.00` ✅ |

TextBlob's pattern-based lexicon misses informal/social-media language. VADER — built specifically for this use case — correctly handles ALL CAPS, `!`, negations (`"won't"`), and slang.

---

## ☁️ Deployment — Streamlit Cloud

Deploy this app publicly in under 5 minutes:

1. **Push to GitHub** — ensure `model/model.pkl` and `model/vectorizer.pkl` are committed
   ```bash
   git add .
   git commit -m "feat: add trained model artefacts"
   git push origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub

3. **Create a new app:**
   - Repository: `YOUR_USERNAME/ai-content-manipulation-detector`
   - Branch: `main`
   - Main file path: `app.py`

4. Click **Deploy** — Streamlit Cloud auto-installs `requirements.txt`

> **Tip:** If you prefer not to commit `.pkl` files, add a Streamlit Cloud startup command that runs `train.py` as a pre-run step using `packages.txt` and a custom `start.sh`.

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas to extend this project:

- [ ] Add more training data (real annotated datasets)
- [ ] Replace Logistic Regression with a fine-tuned `DistilBERT` model
- [ ] Add a REST API endpoint using FastAPI
- [ ] Add multi-language support
- [ ] Integrate explainability (LIME / SHAP feature importance)
- [ ] Add batch analysis (upload a `.csv` of texts)

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [NLTK](https://www.nltk.org/) — Natural Language Toolkit
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) — C.J. Hutto & E.E. Gilbert
- [TextBlob](https://textblob.readthedocs.io/) — Steven Loria
- [scikit-learn](https://scikit-learn.org/) — ML framework
- [Streamlit](https://streamlit.io/) — Web app framework

---

<div align="center">

**Built with ❤️ during the Microsoft Elevate AI Internship · 2026**

*If you found this useful, please consider giving it a ⭐ on GitHub!*

</div>
