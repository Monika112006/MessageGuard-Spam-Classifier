
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
df = df[["v1", "v2"]].copy()
df.columns = ["label", "message"]
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df["label"] = df["label"].map({"ham": 0, "spam": 1})
df.dropna(subset=["label"], inplace=True)
df["label"] = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Test 1: Current configuration
baseline_vectorizer = TfidfVectorizer(stop_words="english")
X_train_base = baseline_vectorizer.fit_transform(X_train)
X_test_base = baseline_vectorizer.transform(X_test)

baseline_model = MultinomialNB()
baseline_model.fit(X_train_base, y_train)
baseline_pred = baseline_model.predict(X_test_base)

# Test 2: Include pairs of consecutive words
bigram_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)
X_train_bigram = bigram_vectorizer.fit_transform(X_train)
X_test_bigram = bigram_vectorizer.transform(X_test)

bigram_model = MultinomialNB()
bigram_model.fit(X_train_bigram, y_train)
bigram_pred = bigram_model.predict(X_test_bigram)

# Compare results
for name, predictions in [
    ("Baseline", baseline_pred),
    ("TF-IDF with Bigrams", bigram_pred)
]:
    print(f"\n--- {name} ---")
    print(f"Accuracy:  {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(f"Spam Precision: {precision_score(y_test, predictions, pos_label=1, zero_division=0) * 100:.2f}%")
    print(f"Spam Recall:    {recall_score(y_test, predictions, pos_label=1, zero_division=0) * 100:.2f}%")
    print(f"Spam F1-score:  {f1_score(y_test, predictions, pos_label=1, zero_division=0) * 100:.2f}%")
