
document.addEventListener("DOMContentLoaded", () => {
    const message = document.getElementById("message");
    const charCount = document.getElementById("charCount");

    function updateCharacterCount() {
        if (message && charCount) {
            charCount.textContent = `${message.value.length} / 5000`;
        }
    }

    if (message) {
        updateCharacterCount();
        message.addEventListener("input", updateCharacterCount);
    }

    const explanations = {
        accuracy: {
            title: "Accuracy",
            description: "The percentage of all test messages classified correctly. Accuracy can hide weaknesses when classes are imbalanced."
        },
        precision: {
            title: "Spam precision",
            description: "Of the messages predicted as spam, the proportion that really were spam. Higher precision means fewer legitimate messages are incorrectly flagged."
        },
        recall: {
            title: "Spam recall",
            description: "Of all actual spam messages, the proportion detected by the model. Higher recall means fewer spam messages are missed."
        },
        f1: {
            title: "Spam F1-score",
            description: "The harmonic mean of spam precision and spam recall. It balances false alarms against missed spam."
        }
    };

    const explanationPanel = document.getElementById("metricExplanation");

    document.querySelectorAll("[data-metric]").forEach(button => {
        button.addEventListener("click", () => {
            const info = explanations[button.dataset.metric];
            if (!info || !explanationPanel) return;

            document.getElementById("metricTitle").textContent = info.title;
            document.getElementById("metricDescription").textContent = info.description;
            explanationPanel.hidden = false;
            explanationPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
        });
    });

    const matrixDescriptions = {
        tn: "True negative: a normal message correctly classified as Ham.",
        fp: "False positive: a normal message incorrectly classified as Spam.",
        fn: "False negative: a spam message incorrectly classified as Ham.",
        tp: "True positive: a spam message correctly classified as Spam."
    };

    const matrixExplanation = document.getElementById("matrixExplanation");

    document.querySelectorAll("[data-matrix]").forEach(button => {
        button.addEventListener("click", () => {
            if (!matrixExplanation) return;
            matrixExplanation.textContent =
                `${matrixDescriptions[button.dataset.matrix]} Count: ${button.dataset.count}.`;
        });
    });

    const comparisonCanvas = document.getElementById("comparisonChart");
    const selector = document.getElementById("metricSelector");
    const datasetCanvas = document.getElementById("datasetChart");

    if (typeof Chart === "undefined") {
        console.warn("Chart.js did not load. Charts are unavailable.");
        return;
    }

    const modelNames = ["SVM", "Naive Bayes", "Logistic Regression"];

    const metrics = {
        accuracy: [97.97, 96.62, 95.84],
        spam_precision: [94, 99, 99],
        spam_recall: [89, 74, 68],
        spam_f1: [92, 85, 81]
    };

    const metricLabels = {
        accuracy: "Accuracy (%)",
        spam_precision: "Spam precision (%)",
        spam_recall: "Spam recall (%)",
        spam_f1: "Spam F1-score (%)"
    };

    let comparisonChart = null;

    function renderComparison(metricName) {
        if (!comparisonCanvas) return;

        if (comparisonChart) comparisonChart.destroy();

        comparisonChart = new Chart(comparisonCanvas, {
            type: "bar",
            data: {
                labels: modelNames,
                datasets: [{
                    label: metricLabels[metricName],
                    data: metrics[metricName],
                    backgroundColor: ["#55d6a0", "#7aa7ff", "#8d82f5"],
                    borderRadius: 8,
                    maxBarThickness: 54
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        ticks: { color: "#9aa9c0" },
                        grid: { color: "#263247" }
                    },
                    x: {
                        ticks: { color: "#9aa9c0" },
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: context => `${context.raw}%`
                        }
                    }
                }
            }
        });
    }

    if (selector) {
        renderComparison(selector.value);
        selector.addEventListener("change", () => {
            renderComparison(selector.value);
        });
    }

    if (datasetCanvas) {
        new Chart(datasetCanvas, {
            type: "doughnut",
            data: {
                labels: ["Ham", "Spam"],
                datasets: [{
                    data: [903, 131],
                    backgroundColor: ["#55d6a0", "#ff7d87"],
                    borderColor: "#111827",
                    borderWidth: 4
                }]
            },
            options: {
                responsive: true,
                cutout: "68%",
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: {
                            color: "#9aa9c0",
                            padding: 18
                        }
                    }
                }
            }
        });
    }
});
