
import pandas as pd
from pathlib import Path

results = Path("results")

lr = pd.read_csv(results / "logistic_regression_results.csv")
svm = pd.read_csv(results / "svm_results.csv")
cnn = pd.read_csv(results / "cnn_results.csv")
rf = pd.read_csv(results / "random_forest_results.csv")

rows = []

# Logistic Regression
for _, r in lr.iterrows():
    rows.append({
        "Model": "Logistic Regression",
        "Features": str(
            r.get("features", "GIST + Color Histogram")
        ).replace("ColorHistogram", "Color Histogram"),
        "Train Accuracy (%)": r["train_accuracy"] * 100,
        "Validation Accuracy (%)": r["validation_accuracy"] * 100,
        "Test Accuracy (%)": r["test_accuracy"] * 100,
        "Training Time (s)": r["training_time_seconds"],
        "Reference Test Accuracy (%)": r["reference_test_accuracy"] * 100
    })

# SVM
for _, r in svm.iterrows():
    rows.append({
        "Model": "SVM",
        "Features": str(r["features"]).replace(
            "ColorHistogram", "Color Histogram"
        ),
        "Train Accuracy (%)": r["train_accuracy"] * 100,
        "Validation Accuracy (%)": r["validation_accuracy"] * 100,
        "Test Accuracy (%)": r["test_accuracy"] * 100,
        "Training Time (s)": r["training_time_seconds"],
        "Reference Test Accuracy (%)": r["reference_test_accuracy"] * 100
    })

# CNN
rows.append({
    "Model": "CNN",
    "Features": "Raw RGB images (256 x 256)",
    "Train Accuracy (%)": cnn.iloc[0]["train_accuracy_percent"],
    "Validation Accuracy (%)": 65.6836450457573,
    "Test Accuracy (%)": cnn.iloc[0]["test_accuracy_percent"],
    "Training Time (s)": None,
    "Reference Test Accuracy (%)": cnn.iloc[0]["reference_test_accuracy_percent"]
})

# Random Forest (extension model)
rows.append({
    "Model": "Random Forest",
    "Features": str(
        "GIST + Color Histogram"
    ).replace("ColorHistogram", "Color Histogram"),
    "Train Accuracy (%)": rf.iloc[0]["train_accuracy_percent"],
    "Validation Accuracy (%)": rf.iloc[0]["validation_accuracy_percent"],
    "Test Accuracy (%)": rf.iloc[0]["test_accuracy_percent"],
    "Training Time (s)": rf.iloc[0]["training_time_seconds"],
    "Reference Test Accuracy (%)": None
})

# Create and save the comparison table
comparison = pd.DataFrame(rows)
comparison.to_csv(
    results / "model_comparison_clean.csv",
    index=False
)

print(comparison.round(2).to_string(index=False))
print("\nSaved to results/model_comparison_clean.csv")

