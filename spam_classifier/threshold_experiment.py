
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

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

vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Probability that each message is spam
spam_probabilities = model.predict_proba(X_test_tfidf)[:, 1]

for threshold in [0.50, 0.40, 0.30, 0.20, 0.10]:
    predictions = (spam_probabilities >= threshold).astype(int)

    print(f"\nThreshold: {threshold:.2f}")
    print(f"Accuracy:  {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0) * 100:.2f}%")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0) * 100:.2f}%")
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0) * 100:.2f}%")
