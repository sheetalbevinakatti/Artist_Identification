
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

# ==================================================
# PROJECT SETTINGS
# ==================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "cnn_artist_identification.keras"
SPLIT_PATH = PROJECT_DIR / "data" / "processed" / "dataset_split.csv"
IMAGES_DIR = PROJECT_DIR / "data" / "raw" / "images"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EXPECTED_CNN_TEST_ACCURACY = 0.6555

st.set_page_config(
    page_title="Artist Identification",
    page_icon="🎨",
    layout="centered",
)

# ==================================================
# LOAD MODEL AND DATA
# ==================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_dataset_split():
    return pd.read_csv(SPLIT_PATH)


@st.cache_data
def load_artist_classes():
    df = load_dataset_split()
    return sorted(df["artist"].dropna().unique().tolist())


def preprocess_image(image):
    """Apply the same image resizing and normalization used for CNN input."""
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32) / 255.0


# ==================================================
# TEST-SET EVALUATION
# ==================================================
@st.cache_data(show_spinner=False)
def evaluate_test_set(model_path, model_mtime, split_path, split_mtime):
    # Load a separate model instance for evaluation.
    # This prevents evaluation from changing the prediction model.
    evaluation_model = tf.keras.models.load_model(model_path)

    df = pd.read_csv(split_path)

    required_columns = {"new_filename", "artist", "split"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "The split CSV must contain new_filename, artist, and split."
        )

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    test_df = df[df["split"] == "test"].copy()

    if test_df.empty:
        raise ValueError("No test images were found.")

    classes = sorted(df["artist"].dropna().unique().tolist())

    if evaluation_model.output_shape[-1] != len(classes):
        raise ValueError(
            "The number of CNN outputs does not match the number of artists."
        )

    class_to_index = {artist: i for i, artist in enumerate(classes)}

    y_true = []
    y_pred = []
    images = []
    labels = []

    for row in test_df.itertuples(index=False):
        image_path = Path(IMAGES_DIR) / str(row.new_filename)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing test image: {image_path}"
            )

        with Image.open(image_path) as image:
            images.append(preprocess_image(image))

        labels.append(class_to_index[row.artist])

        if len(images) == BATCH_SIZE:
            batch_predictions = evaluation_model.predict(
                np.stack(images),
                verbose=0,
            )
            y_pred.extend(np.argmax(batch_predictions, axis=1).tolist())
            y_true.extend(labels)
            images, labels = [], []

    if images:
        batch_predictions = evaluation_model.predict(
            np.stack(images),
            verbose=0,
        )
        y_pred.extend(np.argmax(batch_predictions, axis=1).tolist())
        y_true.extend(labels)

    label_ids = list(range(len(classes)))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true, y_pred,
        labels=label_ids,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true, y_pred,
        labels=label_ids,
        average="weighted",
        zero_division=0,
    )

    precision, recall, f1_values, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=label_ids,
            zero_division=0,
        )
    )

    per_artist = pd.DataFrame({
        "Artist": classes,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1_values,
        "Support": support,
    })

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=label_ids,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.index.name = "Class"

    return (
        accuracy,
        macro_f1,
        weighted_f1,
        per_artist,
        matrix,
        classes,
        report_df,
        len(y_true),
    )


# ==================================================
# PAGE HEADER
# ==================================================
st.title("🎨 Artist Identification")
st.caption(
    "Machine learning for classifying paintings into 15 artist categories."
)

if not MODEL_PATH.exists():
    st.error(f"Saved CNN model not found: {MODEL_PATH}")
    st.stop()

if not SPLIT_PATH.exists():
    st.error(f"Dataset split file not found: {SPLIT_PATH}")
    st.stop()

try:
    model = load_model()
    artist_classes = load_artist_classes()
except Exception as exc:
    st.error(f"Could not load the model or labels: {exc}")
    st.stop()

if len(artist_classes) != model.output_shape[-1]:
    st.error(
        "The number of artist labels does not match the model outputs. "
        "Please verify the training label mapping."
    )
    st.stop()

# ==================================================
# TABS
# ==================================================
prediction_tab, evaluation_tab = st.tabs(
    ["🖼️ Predict Artist", "📊 Model Evaluation"]
)

# ==================================================
# TAB 1: PREDICTION
# ==================================================
with prediction_tab:
    st.subheader("Upload a painting")

    uploaded_file = st.file_uploader(
        "Choose a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        key="painting_upload",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            left, right = st.columns([1, 1])

            with left:
                st.image(
                    image,
                    caption="Uploaded painting",
                    use_container_width=True,
                )

            image_array = preprocess_image(image)
            input_batch = np.expand_dims(image_array, axis=0)

            with st.spinner("Analyzing painting..."):
                probabilities = model.predict(
                    input_batch,
                    verbose=0,
                )[0]

            top_indices = np.argsort(probabilities)[::-1][:3]
            best_index = int(top_indices[0])

            with right:
                st.success(
                    f"Predicted artist: {artist_classes[best_index]}"
                )

                st.metric(
                    "Model probability",
                    f"{probabilities[best_index] * 100:.2f}%",
                )

                st.caption(
                    "This is the model's predicted probability, "
                    "not a guarantee that the prediction is correct."
                )

                st.markdown("#### Top 3 predictions")

                for index in top_indices:
                    artist = artist_classes[int(index)]
                    probability = float(probabilities[index])

                    st.write(
                        f"**{artist}** — {probability * 100:.2f}%"
                    )
                    st.progress(probability)

        except Exception as exc:
            st.error(f"Could not process this image: {exc}")

    else:
        st.info("Upload a painting to see the predicted artist.")

# ==================================================
# TAB 2: EVALUATION DASHBOARD
# ==================================================
with evaluation_tab:
    st.subheader("CNN Test-Set Evaluation")

    st.write(
        "Evaluate the saved CNN against the actual artist labels "
        "for all paintings in the held-out test set."
    )

    st.caption(
        "This is a separate evaluation from the single-image prediction. "
        "The CNN is not retrained."
    )

    if not IMAGES_DIR.exists():
        st.error(
            f"Test image directory not found: {IMAGES_DIR}"
        )
    else:
        if st.button(
            "▶ Run Test-Set Evaluation",
            type="primary",
        ):
            st.session_state["run_evaluation"] = True

        if st.session_state.get("run_evaluation", False):
            try:
                with st.spinner(
                    "Evaluating test images. The first run may take a while..."
                ):
                    results = evaluate_test_set(
                        str(MODEL_PATH),
                        MODEL_PATH.stat().st_mtime,
                        str(SPLIT_PATH),
                        SPLIT_PATH.stat().st_mtime,
                    )

                (
                    accuracy,
                    macro_f1,
                    weighted_f1,
                    per_artist,
                    matrix,
                    classes,
                    report_df,
                    test_count,
                ) = results

                st.success(
                    f"Evaluation completed on {test_count} test paintings."
                )

                # ----------------------------------
                # OVERALL METRICS
                # ----------------------------------
                st.markdown("### Overall Performance")

                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)

                col1.metric(
                    "Test Accuracy",
                    f"{accuracy * 100:.2f}%",
                )
                col2.metric(
                    "Macro F1-score",
                    f"{macro_f1:.4f}",
                )
                col3.metric(
                    "Weighted F1-score",
                    f"{weighted_f1:.4f}",
                )
                col4.metric(
                    "Test Paintings",
                    f"{test_count}",
                )

                st.caption(
                    "Macro F1 treats each artist equally. Weighted F1 "
                    "accounts for the number of test paintings per artist."
                )

                if abs(accuracy - EXPECTED_CNN_TEST_ACCURACY) > 0.002:
                    st.warning(
                        "The evaluated accuracy differs from the previously "
                        "recorded CNN test accuracy of 65.55%. Verify the "
                        "model file, output-label order, preprocessing, and "
                        "test split before using the results in your report."
                    )

                # ----------------------------------
                # PER-ARTIST GRAPH
                # ----------------------------------
                st.markdown("### Per-Artist Performance")

                plot_df = per_artist.set_index("Artist")[
                    ["Precision", "Recall", "F1-score"]
                ]

                fig, ax = plt.subplots(figsize=(10, 4.5))
                plot_df.plot(kind="bar", ax=ax, width=0.8)

                ax.set_title(
                    "Precision, Recall and F1-score by Artist",
                    fontsize=12,
                )
                ax.set_ylabel("Score")
                ax.set_xlabel("")
                ax.set_ylim(0, 1.05)
                ax.tick_params(axis="x", labelrotation=45, labelsize=8)
                ax.tick_params(axis="y", labelsize=9)
                ax.legend(fontsize=9, loc="lower right")
                ax.grid(axis="y", alpha=0.25)
                ax.set_axisbelow(True)

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                with st.expander("View per-artist metric table"):
                    st.dataframe(
                        per_artist.style.format({
                            "Precision": "{:.3f}",
                            "Recall": "{:.3f}",
                            "F1-score": "{:.3f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                        height=350,
                    )

                # ----------------------------------
                # CONFUSION MATRIX
                # ----------------------------------
                st.markdown("### Confusion Matrix")

                st.write(
                    "Rows represent actual artists; columns represent "
                    "predicted artists. Correct predictions appear along "
                    "the main diagonal."
                )

                normalize_matrix = st.checkbox(
                    "Show normalized confusion matrix",
                    value=False,
                )

                displayed_matrix = matrix.astype(float)

                if normalize_matrix:
                    row_totals = displayed_matrix.sum(
                        axis=1,
                        keepdims=True,
                    )
                    displayed_matrix = np.divide(
                        displayed_matrix,
                        row_totals,
                        out=np.zeros_like(displayed_matrix),
                        where=row_totals != 0,
                    )

                fig, ax = plt.subplots(figsize=(9, 7))

                heatmap = ax.imshow(
                    displayed_matrix,
                    interpolation="nearest",
                    aspect="auto",
                )

                fig.colorbar(
                    heatmap,
                    ax=ax,
                    fraction=0.046,
                    pad=0.04,
                )

                ax.set_xticks(range(len(classes)))
                ax.set_yticks(range(len(classes)))
                ax.set_xticklabels(
                    classes,
                    rotation=60,
                    ha="right",
                    fontsize=8,
                )
                ax.set_yticklabels(classes, fontsize=8)

                ax.set_xlabel("Predicted Artist")
                ax.set_ylabel("Actual Artist")
                ax.set_title("CNN Confusion Matrix")

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                matrix_df = pd.DataFrame(
                    matrix,
                    index=classes,
                    columns=classes,
                )
                matrix_df.index.name = "Actual / Predicted"

                st.download_button(
                    "⬇ Download Confusion Matrix CSV",
                    data=matrix_df.to_csv().encode("utf-8"),
                    file_name="cnn_confusion_matrix.csv",
                    mime="text/csv",
                )

                # ----------------------------------
                # CLASSIFICATION REPORT
                # ----------------------------------
                st.markdown("### Classification Report")

                with st.expander("View detailed classification report"):
                    st.dataframe(
                        report_df.style.format({
                            column: "{:.4f}"
                            for column in report_df.select_dtypes(
                                include="number"
                            ).columns
                        }),
                        use_container_width=True,
                    )

                st.download_button(
                    "⬇ Download Classification Report CSV",
                    data=report_df.to_csv().encode("utf-8"),
                    file_name="cnn_classification_report.csv",
                    mime="text/csv",
                )

            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                st.caption(
                    "Check that the saved model, dataset split, test images, "
                    "and required packages are available."
                )
