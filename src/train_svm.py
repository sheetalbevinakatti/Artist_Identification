import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC


FEATURE_DIR = Path("data/processed/svm_features")
SPLIT_FILE = Path("data/processed/dataset_split.csv")
RESULT_DIR = Path("results")

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# Load split information
split_df = pd.read_csv(SPLIT_FILE)
y = split_df["artist"].values

train_mask = split_df["split"] == "train"
val_mask = split_df["split"] == "validation"
test_mask = split_df["split"] == "test"

print(f"Train      : {train_mask.sum()}")
print(f"Validation : {val_mask.sum()}")
print(f"Test       : {test_mask.sum()}")


def train_svm(feature_file, feature_name, reference_accuracy):

    print("\n" + "=" * 60)
    print(f"SVM: {feature_name}")
    print("=" * 60)

    X = np.load(FEATURE_DIR / feature_file)

    print(f"Feature shape: {X.shape}")

    X_train = X[train_mask]
    X_val = X[val_mask]
    X_test = X[test_mask]

    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]

    # Min-Max scaling
    print("Applying Min-Max scaling...")

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # PCA: 100 components as described in the paper
    print("Applying PCA (100 components)...")

    pca = PCA(
        n_components=100,
        random_state=42
    )

    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)

    print(f"PCA shape: {X_train.shape}")

    # RBF SVM
    print("Training SVM...")

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        decision_function_shape="ovr",
        random_state=42
    )

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    # Accuracy
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    start_inference = time.perf_counter()

    test_predictions = model.predict(X_test)

    inference_time = time.perf_counter() - start_inference

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    print("\nResults:")
    print(f"Train accuracy      : {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy : {val_accuracy * 100:.2f}%")
    print(f"Test accuracy       : {test_accuracy * 100:.2f}%")
    print(f"Training time       : {training_time:.2f} seconds")
    print(f"Inference time      : {inference_time:.4f} seconds")
    print(f"Reference accuracy  : {reference_accuracy * 100:.1f}%")

    return {
        "model": "SVM",
        "features": feature_name,
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "reference_test_accuracy": reference_accuracy
    }


# ============================================================
# PAPER'S THREE REPORTED SVM CONFIGURATIONS
# ============================================================

results = []

results.append(
    train_svm(
        "gist_hu.npy",
        "GIST + Hu Moments",
        0.681
    )
)

results.append(
    train_svm(
        "gist_color.npy",
        "GIST + Color Histogram",
        0.673
    )
)

results.append(
    train_svm(
        "gist_color_hu.npy",
        "GIST + Color Histogram + Hu Moments",
        0.612
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_file = RESULT_DIR / "svm_results.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("ALL SVM EXPERIMENTS COMPLETE")
print("=" * 60)

print(results_df.to_string(index=False))

print(f"\nResults saved to: {output_file}")