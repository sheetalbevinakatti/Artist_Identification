
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "processed" / "svm_features"
SPLIT_PATH = ROOT / "data" / "processed" / "dataset_split.csv"
RESULTS_DIR = ROOT / "results"

# Load the feature matrix and its metadata
X = np.load(FEATURE_DIR / "gist_color.npy")
metadata = pd.read_csv(FEATURE_DIR / "metadata.csv")
splits = pd.read_csv(SPLIT_PATH)

# Match each feature row to the corresponding split using filename + artist
data = metadata.merge(
    splits,
    left_on=["filename", "artist"],
    right_on=["new_filename", "artist"],
    how="left",
    validate="one_to_one"
)

if len(data) != len(metadata) or data["split_y"].isna().any():
    raise ValueError("Feature metadata could not be matched to every dataset split.")

if len(X) != len(data):
    raise ValueError("Feature rows and metadata rows do not match.")

# Verify split assignments agree
if not (data["split_x"] == data["split_y"]).all():
    raise ValueError("Metadata split assignments differ from dataset_split.csv.")

# Use consistent artist labels
encoder = LabelEncoder()
y = encoder.fit_transform(data["artist"])

train_mask = data["split_y"].eq("train").to_numpy()
val_mask = data["split_y"].eq("validation").to_numpy()
test_mask = data["split_y"].eq("test").to_numpy()

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print("Feature shape:", X.shape)
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)
print("Number of artists:", len(encoder.classes_))

# Random Forest is the extension model; it is not one of the
# three principal models evaluated in the reference paper.
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

start = time.perf_counter()
model.fit(X_train, y_train)
training_seconds = time.perf_counter() - start

train_accuracy = accuracy_score(y_train, model.predict(X_train))
val_accuracy = accuracy_score(y_val, model.predict(X_val))

start = time.perf_counter()
test_predictions = model.predict(X_test)
inference_seconds = time.perf_counter() - start

test_accuracy = accuracy_score(y_test, test_predictions)

print(f"\nTraining accuracy:   {train_accuracy * 100:.2f}%")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
print(f"Test accuracy:       {test_accuracy * 100:.2f}%")
print(f"Training time:       {training_seconds:.2f} seconds")
print(f"Test prediction time:{inference_seconds:.4f} seconds")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame([{
    "model": "Random Forest",
    "features": "GIST + Color Histogram",
    "train_accuracy_percent": train_accuracy * 100,
    "validation_accuracy_percent": val_accuracy * 100,
    "test_accuracy_percent": test_accuracy * 100,
    "training_time_seconds": training_seconds,
    "test_inference_time_seconds": inference_seconds,
    "n_estimators": 200,
    "random_state": 42
}]).to_csv(RESULTS_DIR / "random_forest_results.csv", index=False)

print("\nResults saved to results/random_forest_results.csv")
