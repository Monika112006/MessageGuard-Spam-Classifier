
import csv
import io
import pickle
import re
from pathlib import Path

from flask import (
    Flask, render_template, request,
    jsonify, Response, send_file
)
from markupsafe import Markup, escape
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="templates",
    static_url_path="/static"
)
BASE_DIR = Path(__file__).resolve().parent

# Load the existing trained model and vectorizer.
with open(BASE_DIR / "svm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

SUSPICIOUS_WORDS = {
    "urgent", "free", "claim", "winner", "won", "prize",
    "lottery", "congratulations", "cash", "click", "offer",
    "limited", "immediately", "guaranteed", "reward"
}

# These values come from the evaluation output you shared.
# Update them whenever you evaluate a new model version.
EVALUATION = {
    "test_size": 1034,
    "ham_support": 903,
    "spam_support": 131,
    "models": {
        "SVM": {
            "accuracy": 97.97,
            "ham_precision": 0.98,
            "ham_recall": 0.99,
            "ham_f1": 0.99,
            "spam_precision": 0.94,
            "spam_recall": 0.89,
            "spam_f1": 0.92,
            "matrix": [[896, 7], [14, 117]]
        },
        "Naive Bayes": {
            "accuracy": 96.62,
            "ham_precision": 0.96,
            "ham_recall": 1.00,
            "ham_f1": 0.98,
            "spam_precision": 0.99,
            "spam_recall": 0.74,
            "spam_f1": 0.85,
            "matrix": [[902, 1], [34, 97]]
        },
        "Logistic Regression": {
            "accuracy": 95.84,
            "ham_precision": 0.96,
            "ham_recall": 1.00,
            "ham_f1": 0.98,
            "spam_precision": 0.99,
            "spam_recall": 0.68,
            "spam_f1": 0.81,
            "matrix": [[902, 1], [42, 89]]
        }
    }
}


def highlight_suspicious_words(message):
    pattern = re.compile(
        r"\b(" + "|".join(
            re.escape(word)
            for word in sorted(SUSPICIOUS_WORDS, key=len, reverse=True)
        ) + r")\b",
        re.IGNORECASE
    )

    parts = []
    last = 0

    for match in pattern.finditer(message):
        parts.append(escape(message[last:match.start()]))
        parts.append(
            Markup('<mark class="suspicious-word">') +
            escape(match.group()) +
            Markup("</mark>")
        )
        last = match.end()

    parts.append(escape(message[last:]))
    return Markup("").join(parts)


def get_prediction(message):
    transformed = vectorizer.transform([message])
    prediction = int(model.predict(transformed)[0])

    # The saved SVM is calibrated and supports predict_proba().
    probabilities = model.predict_proba(transformed)[0]
    classes = list(model.classes_)

    spam_index = classes.index(1)
    ham_index = classes.index(0)

    spam_probability = float(probabilities[spam_index])
    ham_probability = float(probabilities[ham_index])

    if spam_probability >= 0.80:
        label = "Spam"
        explanation = "The model estimates a high likelihood of spam."
    elif spam_probability >= 0.50:
        label = "Slightly Likely to Be Spam"
        explanation = "The model detects some spam-like patterns."
    elif ham_probability > 0.80:
        label = "Ham"
        explanation = "The model estimates a high likelihood of a normal message."
    else:
        label = "Uncertain"
        explanation = "The model's confidence is not strong enough for a clear result."

    return {
        "label": label,
        "prediction": "Spam" if prediction == 1 else "Ham",
        "spam_probability": round(spam_probability * 100, 2),
        "ham_probability": round(ham_probability * 100, 2),
        "explanation": explanation
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            try:
                result = get_prediction(message)
                result["highlighted_message"] = highlight_suspicious_words(message)
            except Exception:
                app.logger.exception("Message prediction failed")
                result = {
                    "label": "Error",
                    "explanation": "Unable to analyze this message. Please try again."
                }

    return render_template(
        "index.html",
        message=message,
        result=result,
        evaluation=EVALUATION
    )


@app.route("/api/evaluation")
def evaluation_api():
    """Return saved evaluation results for dashboard charts."""
    return jsonify(EVALUATION)


@app.route("/download/evaluation.csv")
def download_csv():
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Model", "Accuracy (%)", "Ham Precision", "Ham Recall",
        "Ham F1", "Spam Precision", "Spam Recall", "Spam F1",
        "TN", "FP", "FN", "TP"
    ])

    for name, metrics in EVALUATION["models"].items():
        tn, fp = metrics["matrix"][0]
        fn, tp = metrics["matrix"][1]

        writer.writerow([
            name, metrics["accuracy"],
            metrics["ham_precision"], metrics["ham_recall"],
            metrics["ham_f1"], metrics["spam_precision"],
            metrics["spam_recall"], metrics["spam_f1"],
            tn, fp, fn, tp
        ])

    response = Response(
        output.getvalue(),
        mimetype="text/csv"
    )
    response.headers["Content-Disposition"] = (
        "attachment; filename=messageguard_evaluation.csv"
    )
    return response


@app.route("/download/evaluation.pdf")
def download_pdf():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        )
    except ImportError:
        return (
            "PDF export requires reportlab. Install it using "
            "'pip install reportlab' and restart the app.",
            500
        )

    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("MessageGuard — Model Evaluation Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(
            "Saved test-set results supplied by the training script. "
            "These are not a live evaluation of newly submitted messages.",
            styles["BodyText"]
        ),
        Spacer(1, 16)
    ]

    data = [[
        "Model", "Accuracy", "Spam Precision", "Spam Recall", "Spam F1"
    ]]

    for name, metrics in EVALUATION["models"].items():
        data.append([
            name,
            f'{metrics["accuracy"]:.2f}%',
            f'{metrics["spam_precision"]:.2f}',
            f'{metrics["spam_recall"]:.2f}',
            f'{metrics["spam_f1"]:.2f}'
        ])

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))
    story.append(table)
    story.append(Spacer(1, 18))

    svm = EVALUATION["models"]["SVM"]
    story.append(Paragraph("SVM Confusion Matrix", styles["Heading2"]))

    tn, fp = svm["matrix"][0]
    fn, tp = svm["matrix"][1]
    matrix = [
        ["Actual / Predicted", "Ham", "Spam"],
        ["Ham", str(tn), str(fp)],
        ["Spam", str(fn), str(tp)]
    ]

    matrix_table = Table(matrix)
    matrix_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f1")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8)
    ]))
    story.append(matrix_table)

    document.build(story)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="messageguard_evaluation.pdf"
    )


@app.route("/sender-analyzer", methods=["GET", "POST"])
def sender_analyzer():
    status = None
    phone = ""

    if request.method == "POST":
        phone = request.form.get("phone", "").strip()

        if not phone:
            status = "skipped"
        elif not re.fullmatch(r"[+0-9\s()-]+", phone):
            status = "invalid"
        else:
            digits = re.sub(r"\D", "", phone)

            if len(digits) == 12 and digits.startswith("91"):
                digits = digits[2:]

            status = (
                "valid"
                if re.fullmatch(r"[6-9]\d{9}", digits)
                else "invalid"
            )

    return render_template(
        "sender_analyzer.html",
        phone=phone,
        status=status
    )


if __name__ == "__main__":
    app.run(debug=True)
