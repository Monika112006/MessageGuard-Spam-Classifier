
import pickle

# Load the trained model
with open("spam_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Classification threshold
THRESHOLD = 0.20

print("=== SMS Spam Classifier ===")
print("Enter a message to check whether it is spam.")
print("Type 'exit' to stop.\n")

while True:
    message = input("Enter your message: ").strip()

    if message.lower() == "exit":
        print("Classifier closed.")
        break

    if not message:
        print("Please enter a message.\n")
        continue

    # Convert the message into TF-IDF features
    vectorized_text = vectorizer.transform([message])

    # Get the probability that the message is spam
    spam_probability = model.predict_proba(vectorized_text)[0][1]

    # Classify using the 0.20 threshold
    if spam_probability >= THRESHOLD:
        print("Prediction: SPAM")
    else:
        print("Prediction: NOT SPAM")

    print(f"Spam probability: {spam_probability:.2%}")
    print("-" * 35)
