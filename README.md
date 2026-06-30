# 📧 Spam Mail Detector

**QSkill AI/ML Internship Task — June/July 2026**

A complete, beginner-friendly spam classifier built with Python.
Uses the **SMS Spam Collection (UCI)** dataset and compares multiple models.

---

## 🎯 Objective

Build a classifier that distinguishes **spam** from **ham (not spam)** messages
using classical NLP and ML techniques.

---

## 🗂️ Project Structure

```
spam_detector/
├── spam_detector.py     ← Main script (all-in-one pipeline)
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
├── data/                ← Dataset downloaded automatically at runtime
│   └── SMSSpamCollection
└── results/             ← Charts & confusion matrices (generated at runtime)
    ├── label_distribution.png
    ├── cm_naive_bayes_bow.png
    ├── cm_naive_bayes_tfidf.png
    ├── cm_logistic_regression_tfidf.png
    └── model_comparison.png
```

---

## ⚙️ Setup & Run

### 1. Clone / Download

```bash
git clone https://github.com/anik-chaudhary/spam-detector.git
cd spam-detector
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python spam_detector.py
```

The dataset is **downloaded automatically** from the UCI ML Repository on first run.
If the download fails (e.g. no internet), a sample dataset is created automatically.

---

## 🔄 Pipeline Steps

| Step | Description                                                         |
| ---- | ------------------------------------------------------------------- |
| 1    | Download SMS Spam Collection (UCI)                                  |
| 2    | Load & explore data (~5,574 messages)                               |
| 3    | Preprocess: lowercase → remove URLs/digits/punct → stopword removal |
| 4    | Feature extraction: Bag-of-Words + TF-IDF (with bigrams)            |
| 5    | Train/Test split (80/20, stratified)                                |
| 6    | Train 3 models: NB+BoW, NB+TF-IDF, LogisticRegression+TF-IDF        |
| 7    | Evaluate: Accuracy, Precision, Recall, F1 Score                     |
| 8    | Save confusion matrices + comparison chart to `results/`            |
| 9    | Interactive demo mode                                               |

---

## 📊 Expected Results

| Model                            | Accuracy | F1 (Spam) |
| -------------------------------- | -------- | --------- |
| Naive Bayes (BoW)                | ~97%     | ~0.94     |
| Naive Bayes (TF-IDF)             | ~97%     | ~0.94     |
| **Logistic Regression (TF-IDF)** | **~98%** | **~0.96** |

---

## 🧠 Skills Demonstrated

- Text preprocessing & tokenization (zero external NLP deps)
- Bag-of-Words & TF-IDF feature extraction with bigrams
- Naive Bayes & Logistic Regression classifiers
- Evaluation: accuracy, precision, recall, F1, confusion matrix
- Matplotlib + Seaborn visualizations
- Interactive prediction mode

---

## 📦 Dataset

**SMS Spam Collection** — UCI ML Repository
🔗 https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
5,574 English SMS messages labelled as `spam` or `ham`.

---

## 📜 License

MIT — Free to use for learning and internship purposes.
