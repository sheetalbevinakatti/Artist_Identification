import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = Path("data/raw/images")
SPLIT_FILE = Path("data/processed/dataset_split.csv")
RESULT_DIR = Path("results")

RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (256, 256)
BATCH_SIZE = 96
EPOCHS = 30
SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOAD SPLIT
# ============================================================

print("Loading dataset split...")

df = pd.read_csv(SPLIT_FILE)

print(f"Total images: {len(df)}")


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(df["artist"])

print(f"Number of artists: {len(label_encoder.classes_)}")


# ============================================================
# IMAGE PATHS
# ============================================================

df["image_path"] = df["new_filename"].apply(
    lambda x: str(IMAGE_DIR / x)
)


# ============================================================
# CREATE DATASETS
# ============================================================

def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image, label


def create_dataset(dataframe, shuffle=False):

    paths = dataframe["image_path"].values
    labels = dataframe["label"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=SEED
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

print("\nDataset sizes:")
print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")


train_ds = create_dataset(
    train_df,
    shuffle=True
)

val_ds = create_dataset(
    val_df
)

test_ds = create_dataset(
    test_df
)


# ============================================================
# DATA AUGMENTATION
# Paper mentions rotation, zoom, flipping and shearing.
# ============================================================

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.1
    ),

    tf.keras.layers.RandomZoom(
        0.1
    ),
])


# ============================================================
# CNN MODEL
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(256, 256, 3)
    ),

    augmentation,

    # Conv block 1
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Conv block 2
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Conv block 3
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Conv block 4
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Conv block 5
    tf.keras.layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Conv block 6
    tf.keras.layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        kernel_initializer="he_normal"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    # Dense layers
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        512,
        kernel_initializer="he_normal"
    ),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Dropout(
        0.5
    ),

    tf.keras.layers.Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.000074,
        beta_1=0.9,
        beta_2=0.999
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# TRAIN
# ============================================================

print("\nStarting CNN training...")

start_time = time.perf_counter()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

training_time = time.perf_counter() - start_time


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\nEvaluating on test set...")

test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)


# ============================================================
# TRAINING ACCURACY
# ============================================================

train_accuracy = history.history["accuracy"][-1]
validation_accuracy = history.history["val_accuracy"][-1]


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CNN RESULTS")
print("=" * 60)

print(
    f"Train accuracy      : "
    f"{train_accuracy * 100:.2f}%"
)

print(
    f"Validation accuracy : "
    f"{validation_accuracy * 100:.2f}%"
)

print(
    f"Test accuracy       : "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Training time       : "
    f"{training_time:.2f} seconds"
)

print(
    f"Reference accuracy  : 74.7%"
)

print("=" * 60)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    "models/cnn_artist_identification.keras"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "model": "CNN",
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "training_time_seconds": training_time,
        "reference_test_accuracy": 0.747
    }
])

results.to_csv(
    RESULT_DIR / "cnn_results.csv",
    index=False
)

print(
    "\nResults saved to: "
    "results/cnn_results.csv"
)