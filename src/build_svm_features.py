import os
import numpy as np
import pandas as pd
from pathlib import Path

from extract_features import extract_features


# --------------------------------------------------
# Paths
# --------------------------------------------------

SPLIT_FILE = "data/processed/dataset_split.csv"
IMAGE_DIR = Path("data/raw/images")
OUTPUT_DIR = Path("data/processed/svm_features")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load dataset split
# --------------------------------------------------

df = pd.read_csv(SPLIT_FILE)

print(f"Total images: {len(df)}")
print(f"Image folder: {IMAGE_DIR}")
print("-" * 60)


# --------------------------------------------------
# Storage for features
# --------------------------------------------------

gist_features = []
color_features = []
hu_features = []

labels = []
filenames = []
splits = []


# --------------------------------------------------
# Extract features
# --------------------------------------------------

for index, row in df.iterrows():

    filename = row["new_filename"]
    image_path = IMAGE_DIR / filename

    try:

        features = extract_features(image_path)

        gist_features.append(features["gist"])
        color_features.append(features["color_histogram"])
        hu_features.append(features["hu_moments"])

        labels.append(row["artist"])
        filenames.append(filename)
        splits.append(row["split"])

    except Exception as error:

        print(f"ERROR: {filename}")
        print(error)


    # Progress message
    if (index + 1) % 250 == 0:
        print(
            f"Processed "
            f"{index + 1}/{len(df)} images"
        )


# --------------------------------------------------
# Convert to NumPy arrays
# --------------------------------------------------

gist_features = np.array(
    gist_features,
    dtype=np.float32
)

color_features = np.array(
    color_features,
    dtype=np.float32
)

hu_features = np.array(
    hu_features,
    dtype=np.float32
)


# --------------------------------------------------
# Create feature combinations
# --------------------------------------------------

gist_hu = np.concatenate(
    [gist_features, hu_features],
    axis=1
)

gist_color = np.concatenate(
    [gist_features, color_features],
    axis=1
)

gist_color_hu = np.concatenate(
    [
        gist_features,
        color_features,
        hu_features
    ],
    axis=1
)


# --------------------------------------------------
# Save features
# --------------------------------------------------

np.save(
    OUTPUT_DIR / "gist_hu.npy",
    gist_hu
)

np.save(
    OUTPUT_DIR / "gist_color.npy",
    gist_color
)

np.save(
    OUTPUT_DIR / "gist_color_hu.npy",
    gist_color_hu
)


# --------------------------------------------------
# Save labels and metadata
# --------------------------------------------------

metadata = pd.DataFrame({
    "filename": filenames,
    "artist": labels,
    "split": splits
})

metadata.to_csv(
    OUTPUT_DIR / "metadata.csv",
    index=False
)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print()
print("=" * 60)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 60)

print(f"Images processed: {len(metadata)}")

print()
print("Feature dimensions:")

print(
    "GIST + Hu:",
    gist_hu.shape
)

print(
    "GIST + Color:",
    gist_color.shape
)

print(
    "GIST + Color + Hu:",
    gist_color_hu.shape
)

print()
print("Saved to:")
print(OUTPUT_DIR)