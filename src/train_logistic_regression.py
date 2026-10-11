import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# PATHS
# ============================================================

FEATURE_DIR = Path("data/processed/svm_features")
SPLIT_FILE = Path("data/processed/dataset_split.csv")
RESULT_DIR = Path("results")

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset split...")
split_df = pd.read_csv(SPLIT_FILE)

print(f"Total samples: {len(split_df)}")


# ============================================================
# LOAD GIST + COLOR FEATURES
# Paper's Logistic Regression baseline
# ============================================================

print("\nLoading GIST + Color features...")

X = np.load(
    FEATURE_DIR / "gist_color.npy"
)

print(f"Feature matrix shape: {X.shape}")


# ============================================================
# LABELS
# ============================================================

y = split_df["artist"].values


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_mask = split_df["split"] == "train"
val_mask = split_df["split"] == "validation"
test_mask = split_df["split"] == "test"

X_train = X[train_mask]
X_val = X[val_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_val = y[val_mask]
y_test = y[test_mask]

print("\nSplit sizes:")
print(f"Train      : {len(X_train)}")
print(f"Validation : {len(X_val)}")
print(f"Test       : {len(X_test)}")


# ============================================================
# MIN-MAX SCALING
# ============================================================

print("\nApplying Min-Max scaling...")

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=2000,
    solver="lbfgs",
    random_state=42
)
start_time = time.perf_counter()

model.fit(X_train, y_train)

training_time = time.perf_counter() - start_time


# ============================================================
# TRAIN ACCURACY
# ============================================================

train_predictions = model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)


# ============================================================
# VALIDATION ACCURACY
# ============================================================

val_predictions = model.predict(X_val)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)


# ============================================================
# TEST ACCURACY
# ============================================================

start_inference = time.perf_counter()

test_predictions = model.predict(X_test)

inference_time = time.perf_counter() - start_inference

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(f"Train accuracy      : {train_accuracy * 100:.2f}%")
print(f"Validation accuracy : {val_accuracy * 100:.2f}%")
print(f"Test accuracy       : {test_accuracy * 100:.2f}%")
print(f"Training time       : {training_time:.2f} seconds")
print(f"Inference time      : {inference_time:.4f} seconds")

print("=" * 60)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "model": "Logistic Regression",
        "features": "GIST + Color Histogram",
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "reference_test_accuracy": 0.612
    }
])

output_file = RESULT_DIR / "logistic_regression_results.csv"

results.to_csv(
    output_file,
    index=False
)

print(f"\nResults saved to: {output_file}")