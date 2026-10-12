
import pickle
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent


# Step 1: Load and clean the dataset
df = pd.read_csv(BASE_DIR / "spam.csv", encoding="latin-1")
df = df[["v1", "v2"]].copy()
df.columns = ["label", "message"]

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df["label"] = df["label"].map({"ham": 0, "spam": 1})
df.dropna(subset=["label"], inplace=True)
df["label"] = df["label"].astype(int)

X = df["message"]
y = df["label"]


# Step 2: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Step 3: Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Step 4: Define the three classification algorithms
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": CalibratedClassifierCV(
        estimator=LinearSVC(),
        method="sigmoid",
        cv=5
    )
}


results = {}


# Step 5: Train and evaluate each model
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print(f"\n{'=' * 45}")
    print(f"MODEL: {name}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Ham", "Spam"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))


# Step 6: Display model comparison
print("\n========== MODEL COMPARISON ==========")

for name, accuracy in sorted(
    results.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{name}: {accuracy * 100:.2f}% accuracy")


# Step 7: Save the Naive Bayes model for compatibility
with open(BASE_DIR / "spam_model.pkl", "wb") as file:
    pickle.dump(models["Naive Bayes"], file)


# Step 8: Save the shared TF-IDF vectorizer
with open(BASE_DIR / "vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)


# Step 9: Save Logistic Regression separately
with open(BASE_DIR / "logistic_regression_model.pkl", "wb") as file:
    pickle.dump(models["Logistic Regression"], file)


# Step 10: Save the calibrated SVM
with open(BASE_DIR / "svm_model.pkl", "wb") as file:
    pickle.dump(models["SVM"], file)


print("\nAll three models trained successfully!")
print("Naive Bayes model saved as spam_model.pkl")
print("Logistic Regression saved as logistic_regression_model.pkl")
print("Calibrated SVM saved as svm_model.pkl")
print("Vectorizer saved as vectorizer.pkl")
print("\nSVM now supports predict_proba() for confidence estimates.")
