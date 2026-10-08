import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/processed/artist_metadata.csv"
OUTPUT_FILE = "data/processed/dataset_split.csv"

df = pd.read_csv(INPUT_FILE)

train_parts = []
val_parts = []
test_parts = []

for artist, group in df.groupby("artist"):

    # 80% train, 20% temporary
    train, temp = train_test_split(
        group,
        test_size=0.20,
        random_state=42
    )

    # Split remaining 20% into 10% validation and 10% test
    validation, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=42
    )

    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"

    train_parts.append(train)
    val_parts.append(validation)
    test_parts.append(test)

result = pd.concat(
    train_parts + val_parts + test_parts,
    ignore_index=True
)

result = result[
    ["new_filename", "artist", "split"]
]

result.to_csv(OUTPUT_FILE, index=False)

print("Dataset split created successfully!")
print()
print(result["split"].value_counts())
print()
print("Split by artist:")
print(pd.crosstab(result["artist"], result["split"]))
print()
print("Total images:", len(result))