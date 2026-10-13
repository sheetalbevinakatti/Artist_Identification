# Artist Identification: Model Documentation

## 1. Project Overview

This project compares machine learning techniques for classifying a
painting into one of 15 predefined artist categories. The selected
dataset contains 7,462 painting images.

Models evaluated: - Logistic Regression - Support Vector Machine (SVM) -
Convolutional Neural Network (CNN) - Random Forest (additional model)

The project follows approaches discussed in the reference paper
*Comparison of Machine Learning Techniques for Artist Identification* by
Jennie Chen and Andrew Deng. Random Forest is included as an extension
to the comparison.

## 2. Dataset and Splitting

  Split          Number of images
  ------------ ------------------
  Training                  5,967
  Validation                  746
  Test                        749
  **Total**             **7,462**

The split is approximately 80:10:10, performed separately for each
artist. The split file is `data/processed/dataset_split.csv`.

The training set fits the models, the validation set supports
development and model assessment, and the test set measures performance
on images not used for training.

## 3. Preprocessing and Feature Extraction

### CNN preprocessing

Images are resized to 256 × 256 pixels, converted to RGB, and normalised
to the range 0--1.

### Features for traditional models

-   **GIST-like features:** The project uses a Gabor filter bank as an
    approximation of a GIST descriptor, producing 512 values per image.
-   **Colour histogram:** 32 values are extracted from each RGB channel,
    giving 96 values per image.
-   **Hu moments:** Seven shape-related descriptor values are extracted
    per image.

  Feature combination                           Number of values per image
  ------------------------------------------- ----------------------------
  GIST-like + Hu moments                                               519
  GIST-like + colour histogram                                         608
  GIST-like + colour histogram + Hu moments                            615

The GIST-like implementation is an approximation, not an exact
reproduction of the original GIST descriptor.

## 4. Model Descriptions

### 4.1 Logistic Regression

Logistic Regression is a supervised classification algorithm. This
project uses GIST-like and colour histogram features with it as a
baseline.

-   Training accuracy: 65.14%
-   Validation accuracy: 58.04%
-   Test accuracy: 57.54%

### 4.2 Support Vector Machine (SVM)

SVM learns decision boundaries to separate classes using numerical
features. Three feature combinations were tested: 1. GIST-like + Hu
moments 2. GIST-like + colour histogram 3. GIST-like + colour
histogram + Hu moments

The recorded pipeline uses MinMax scaling, PCA with 100 components, an
RBF kernel, and a one-vs-rest strategy for 15 classes. The tuning
procedure is simplified compared with the full grid-search procedure
described in the reference paper.

  -----------------------------------------------------------------------
  Feature         Training accuracy         Validation      Test accuracy
  combination                                 accuracy 
  -------------- ------------------ ------------------ ------------------
  GIST-like + Hu             90.51%             47.05%             47.40%
  moments                                              

  GIST-like +                95.42%             57.24%             55.94%
  colour                                               
  histogram                                            

  All three                  96.97%             57.64%             55.94%
  feature types                                        
  -----------------------------------------------------------------------

### 4.3 Convolutional Neural Network (CNN)

A CNN is a deep learning model designed for image tasks. It learns
visual representations directly from RGB painting images.

Recorded architecture and training setup: - Input: 256 × 256 × 3 - Six
convolutional layers with 32, 64, 128, 128, 256, and 256 filters - Five
max-pooling operations - Batch normalisation and ReLU activations -
Flatten layer and Dense layer with 512 units - Dropout rate: 0.5 - Final
softmax layer with 15 outputs - Adam optimiser, learning rate 0.000074 -
Sparse categorical cross-entropy loss - 30 epochs - Augmentation:
horizontal flip, rotation, and zoom

The model was trained in Kaggle and saved as
`models/cnn_artist_identification.keras`.

-   Training accuracy: 99.87%
-   Best validation accuracy: 66.22%
-   Final validation accuracy: 65.68%
-   Test accuracy: 65.55%

### 4.4 Random Forest

Random Forest is an ensemble learning algorithm that combines multiple
decision trees. For classification, the trees' predictions are combined
to produce the final class.

This project uses GIST-like and colour histogram features, 200 trees,
and `random_state=42`. It is the additional model beyond the approaches
in the reference paper.

-   Training accuracy: 100%
-   Validation accuracy: 56.17%
-   Test accuracy: 56.48%
-   Training time: 4.20 seconds

## 5. Results Summary

  Model/configuration                      Test accuracy
  -------------------------------------- ---------------
  Logistic Regression                             57.54%
  SVM --- GIST-like + Hu moments                  47.40%
  SVM --- GIST-like + colour histogram            55.94%
  SVM --- all three feature types                 55.94%
  CNN                                             65.55%
  Random Forest                                   56.48%

Among the recorded experiments, CNN achieved the highest test accuracy
at 65.55%.

Some models have much higher training accuracy than test accuracy. This
gap suggests overfitting: the models fit the training examples more
successfully than they generalise to unseen images.

The reference paper reported 74.7% test accuracy for its CNN. This
project's result is lower. Possible contributors include differences in
dataset splitting, the GIST approximation, SVM tuning, and CNN training
details; the experiment does not establish the individual effect of each
factor.

**Reproducibility note:** The Streamlit evaluation dashboard has also
displayed 64.89% test accuracy, while saved training results record
65.55%. Before reporting one final number, verify the model file,
preprocessing, class order, test split, and evaluation code.

## 6. Streamlit Application

The Streamlit application loads the saved CNN; it does not train the
model when opened. The user uploads a painting, and the application
converts it to RGB, resizes it to 256 × 256, normalises pixel values,
runs inference, and displays the predicted artist and top-three
predictions.

The evaluation section displays metrics and visualisations for the test
set, including precision, recall, F1-score, and a confusion matrix.

Run from the project root:

``` bash
python -m streamlit run src/app.py
```

## 7. Limitations and Future Work

-   The model predicts only among the 15 artist categories used during
    training.
-   Results depend on the dataset, preprocessing, features, and training
    configuration.
-   GIST is approximated using Gabor filters.
-   The training--test accuracy gap suggests room to improve
    generalisation.
-   Future work could explore transfer learning, additional feature
    engineering, and more systematic hyperparameter tuning.

## 8. Maintenance Note

This document describes the recorded implementation and results. Anyone
updating it should verify changes against the current source code and
result files. Commit documentation or code changes under the account of
the teammate who actually makes and reviews those changes.
