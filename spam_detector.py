"""
╔══════════════════════════════════════════════════════════════════╗
║          SPAM MAIL DETECTOR — QSkill AI/ML Internship Task       ║
║                     June/July 2026 Cohort                        ║
╚══════════════════════════════════════════════════════════════════╝

Dataset   : SMS Spam Collection (UCI ML Repository)
Models    : Naive Bayes (MultinomialNB) + Logistic Regression
Features  : Bag-of-Words (CountVectorizer) + TF-IDF
Skills    : Text preprocessing, NLP, feature extraction, classification
"""

# ─────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────────
import os, re, string, urllib.request, zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────────────────────────
# 2. DOWNLOAD DATASET  (SMS Spam Collection — UCI)
# ─────────────────────────────────────────────────────────────────
DATA_DIR  = "data"
DATA_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
ZIP_URL   = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
ZIP_PATH  = os.path.join(DATA_DIR, "smsspamcollection.zip")

def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATA_FILE):
        print(f"[INFO] Dataset already present at '{DATA_FILE}'. Skipping download.")
        return
    print("[INFO] Downloading SMS Spam Collection from UCI …")
    try:
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)
        os.remove(ZIP_PATH)
        print(f"[INFO] Dataset saved to '{DATA_FILE}'.")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("[INFO] Creating a small sample dataset for demo purposes …")
        _create_sample_dataset()

def _create_sample_dataset():
    samples = [
        ("ham",  "Hey, are you free for lunch tomorrow?"),
        ("spam", "FREE entry in 2 a wkly competition to win FA Cup final tkts! Text FA to 87121"),
        ("ham",  "Sounds good, I will see you then."),
        ("spam", "WINNER!! You have been selected as our EXCLUSIVE subscriber. Call 09061702893 now!"),
        ("ham",  "Can you pick up some milk on your way home?"),
        ("spam", "Urgent! Your Mobile number has been awarded a 2000 Bonus Caller Prize. Call 09066368327"),
        ("ham",  "I will be back by 8pm. Dinner ready?"),
        ("spam", "Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed and Free entry 2 100 wkly draw"),
        ("ham",  "Let us catch up this weekend."),
        ("spam", "You have 1 new voicemail. Call 0207 083 6089 now to retrieve. Standard rates apply."),
        ("ham",  "Thanks for the update."),
        ("spam", "SIX chances to win CASH! From 100 to 20000 pounds txt CSH11 and send to 87575"),
        ("ham",  "Are you joining us for the meeting at 3?"),
        ("spam", "IMPORTANT! Please call 0871 872 9925. Calls cost 10p per min."),
        ("ham",  "I finished the assignment, will send it tonight."),
        ("spam", "Win a 1000 cash prize or a prize worth 5000. Call 0906 170 3221"),
        ("ham",  "See you at the gym tomorrow morning."),
        ("spam", "Your free ringtone is waiting to be collected. Simply text the password to 85069."),
        ("ham",  "Please confirm your attendance for the event."),
        ("spam", "Todays vote: Should the government help fight the credit crunch. Text YES or NO to 84080."),
    ] * 30
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        for label, text in samples:
            f.write(f"{label}\t{text}\n")
    print(f"[INFO] Sample dataset written ({len(samples)} rows).")

# ─────────────────────────────────────────────────────────────────
# 3. LOAD DATA
# ─────────────────────────────────────────────────────────────────
def load_data(filepath):
    df = pd.read_csv(filepath, sep="\t", header=None,
                     names=["label", "message"], encoding="latin-1")
    print(f"\n[INFO] Loaded {len(df):,} messages.")
    print(df["label"].value_counts().to_string())
    return df

# ─────────────────────────────────────────────────────────────────
# 4. TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than","too",
    "very","s","t","can","will","just","don","should","now","d","ll","m","o",
    "re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn","haven",
    "isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren","won","wouldn"
}

def preprocess(text):
    """Lowercase → remove URLs → remove digits → remove punctuation → remove stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def add_preprocessed_column(df):
    df = df.copy()
    df["clean_message"] = df["message"].apply(preprocess)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df

# ─────────────────────────────────────────────────────────────────
# 5. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────
def extract_features(X_train, X_test, method="tfidf"):
    """method: 'bow' = Bag-of-Words | 'tfidf' = TF-IDF"""
    if method == "bow":
        vec = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    else:
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)
    print(f"[INFO] Feature matrix — Train: {X_tr.shape}, Test: {X_te.shape}")
    return X_tr, X_te, vec

# ─────────────────────────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, model_name):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n{'='*50}")
    print(f"  Model : {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"{'-'*50}")
    print(classification_report(y_true, y_pred, target_names=["Ham", "Spam"]))
    return {"model": model_name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

# ─────────────────────────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"

def save_confusion_matrix(y_true, y_pred, model_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, f"cm_{model_name.replace(' ','_').replace('(','').replace(')','').lower()}.png")
    plt.savefig(fname, dpi=150); plt.close()
    print(f"[INFO] Saved → {fname}")

def save_metrics_chart(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_r = pd.DataFrame(results)
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(df_r)); width = 0.2
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i*width, df_r[m], width, label=m.capitalize(), color=c, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(df_r["model"], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Spam Detector", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(fname, dpi=150); plt.close()
    print(f"[INFO] Saved → {fname}")

def save_label_distribution(df):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
           colors=["#55A868","#C44E52"], startangle=90,
           wedgeprops={"edgecolor":"white","linewidth":2})
    ax.set_title("Dataset Label Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = os.path.join(RESULTS_DIR, "label_distribution.png")
    plt.savefig(fname, dpi=150); plt.close()
    print(f"[INFO] Saved → {fname}")

# ─────────────────────────────────────────────────────────────────
# 8. PREDICT NEW MESSAGE
# ─────────────────────────────────────────────────────────────────
def predict_message(message, model, vectorizer):
    cleaned  = preprocess(message)
    features = vectorizer.transform([cleaned])
    pred     = model.predict(features)[0]
    prob     = model.predict_proba(features)[0]
    label    = "SPAM" if pred == 1 else "HAM (Not Spam)"
    confidence = prob[pred] * 100
    return f"{label}  (confidence: {confidence:.1f}%)"

# ─────────────────────────────────────────────────────────────────
# 9. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  SPAM MAIL DETECTOR — QSkill AI/ML Internship Task")
    print("="*60)

    # Step 1: Download & Load
    download_dataset()
    df = load_data(DATA_FILE)

    # Step 2: Preprocess
    print("\n[STEP 2] Preprocessing text ...")
    df = add_preprocessed_column(df)
    print(f"  Sample original : {df['message'].iloc[1]}")
    print(f"  Sample cleaned  : {df['clean_message'].iloc[1]}")

    # Step 3: Visualise distribution
    save_label_distribution(df)

    # Step 4: Train/Test split (80/20, stratified)
    X = df["clean_message"]
    y = df["label_num"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[STEP 4] Split — Train: {len(X_train)}, Test: {len(X_test)}")

    results = []

    # Step 5a: Naive Bayes + BoW
    print("\n[STEP 5a] Naive Bayes (Bag-of-Words) ...")
    X_tr_bow, X_te_bow, bow_vec = extract_features(X_train, X_test, "bow")
    nb_bow = MultinomialNB(alpha=0.1)
    nb_bow.fit(X_tr_bow, y_train)
    results.append(evaluate(y_test, nb_bow.predict(X_te_bow), "Naive Bayes (BoW)"))
    save_confusion_matrix(y_test, nb_bow.predict(X_te_bow), "Naive Bayes (BoW)")

    # Step 5b: Naive Bayes + TF-IDF
    print("\n[STEP 5b] Naive Bayes (TF-IDF) ...")
    X_tr_tfidf, X_te_tfidf, tfidf_vec = extract_features(X_train, X_test, "tfidf")
    nb_tfidf = MultinomialNB(alpha=0.1)
    nb_tfidf.fit(X_tr_tfidf, y_train)
    results.append(evaluate(y_test, nb_tfidf.predict(X_te_tfidf), "Naive Bayes (TF-IDF)"))
    save_confusion_matrix(y_test, nb_tfidf.predict(X_te_tfidf), "Naive Bayes (TF-IDF)")

    # Step 5c: Logistic Regression + TF-IDF
    print("\n[STEP 5c] Logistic Regression (TF-IDF) ...")
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    lr.fit(X_tr_tfidf, y_train)
    results.append(evaluate(y_test, lr.predict(X_te_tfidf), "Logistic Regression (TF-IDF)"))
    save_confusion_matrix(y_test, lr.predict(X_te_tfidf), "Logistic Regression (TF-IDF)")

    # Step 6: Comparison chart
    save_metrics_chart(results)

    # Step 7: Demo predictions
    demo_messages = [
        "Congratulations! You've won a FREE iPhone. Click here to claim now!",
        "Hey, are you joining the team meeting at 3 PM today?",
        "URGENT: Your bank account has been compromised. Call 0800-FREE now!",
        "Can you send me the project report by EOD?",
        "Win 1000 cash prize or a trip! Text WIN to 56789 now!",
        "I will be late to the office today, stuck in traffic."
    ]

    print("\n" + "="*60)
    print("  LIVE PREDICTION DEMO  (Best model: Logistic Regression + TF-IDF)")
    print("="*60)
    for msg in demo_messages:
        result = predict_message(msg, lr, tfidf_vec)
        short_msg = (msg[:65] + "...") if len(msg) > 65 else msg
        print(f"\n  Msg    : {short_msg}")
        print(f"  Result : {result}")

    print("\n" + "="*60)
    print("  Done! Charts saved in 'results/' folder.")
    print("="*60 + "\n")

    return lr, tfidf_vec

# ─────────────────────────────────────────────────────────────────
# 10. INTERACTIVE MODE
# ─────────────────────────────────────────────────────────────────
def interactive_mode(model, vectorizer):
    print("\n[INTERACTIVE] Type a message to classify (or 'quit' to exit):\n")
    while True:
        try:
            msg = input("  Your message: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break
        if msg:
            print("  ->", predict_message(msg, model, vectorizer), "\n")

if __name__ == "__main__":
    model, vectorizer = main()
    interactive_mode(model, vectorizer)
