
# Comparison of Machine Learning Techniques for Artist Identification

## 1. Project Overview

This project reproduces selected approaches from the reference paper *Comparison of Machine Learning Techniques for Artist Identification* and extends the experiments using Random Forest.

The objective is to classify a painting into one of 15 artist classes.

## 2. Objectives

- Implement Logistic Regression, SVM, and CNN models from the reference study.
- Extend the study with Random Forest.
- Evaluate models using training, validation, and test accuracy.
- Compare our test results with the reference paper.
- Discuss implementation differences and limitations.

## 3. Dataset

- **Dataset:** Painters by Numbers
- **Selected paintings:** 7,462
- **Number of artist classes:** 15
- **Image dimensions:** 256 x 256 pixels
- **Image format:** RGB
- **Task:** Multiclass artist classification

The images represent paintings by 15 selected artists.

## 4. Models Implemented

### Reference models

1. **Logistic Regression** - baseline classifier using handcrafted image features.
2. **Support Vector Machine (SVM)** - RBF-kernel classifier using combinations of GIST, color histogram, and Hu moment features.
3. **Convolutional Neural Network (CNN)** - image classifier trained on RGB painting images.

### Extension model

4. **Random Forest** - additional model trained using GIST and color histogram features.

**Implementation note:** A Gabor-filter-based approximation was used for GIST features because the reference paper does not provide enough implementation details to reproduce its exact GIST pipeline.

## 5. Evaluation Metrics

- Training accuracy
- Validation accuracy
- Test accuracy
- Training time, where available

## 6. Experimental Results

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 57.54% |
| SVM - GIST + Hu Moments | 47.40% |
| SVM - GIST + Color Histogram | 55.94% |
| SVM - GIST + Color Histogram + Hu Moments | 55.94% |
| CNN | 65.55% |
| Random Forest | 56.48% |

The CNN achieved the highest test accuracy among our experiments, at 65.55%.

The reference paper reported 74.7% test accuracy for its CNN. Our result was lower, potentially because of differences in data splitting, feature extraction, model configuration, and training procedures.

## 7. Project Structure

```text
Artist_Identification_ML/
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|-- notebooks/
|-- reports/
|-- results/
|-- src/
|   |-- build_svm_features.py
|   |-- create_dataset_split.py
|   |-- create_model_comparison.py
|   |-- extract_features.py
|   |-- extract_required_images.py
|   |-- train_cnn.py
|   |-- train_logistic_regression.py
|   |-- train_random_forest.py
|   `-- train_svm.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## 8. Reproducibility

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies using `requirements.txt`.
4. Download the required dataset images and metadata.
5. Place the files in the expected data directories.
6. Run the preprocessing and training scripts as needed.

Large dataset files, generated features, trained models, and result files are excluded from Git tracking by the current `.gitignore` configuration.

## 9. Limitations

- The exact GIST implementation from the reference paper could not be fully reproduced.
- The implemented SVM configuration does not include the full hyperparameter search described in the paper.
- Our dataset split differs slightly from the reference split.
- CNN training was performed in Kaggle, while the other experiments were run locally.
- CNN training time was not recorded in a comparable form and is therefore omitted from the results table.

## 10. Conclusion

This project evaluates three approaches from the reference paper and extends the comparison with Random Forest. The CNN achieved the highest test accuracy in our experiments at 65.55%, followed by Random Forest at 56.48%.

The results demonstrate the performance of the implemented models on the selected painting dataset. Differences from the reference paper should be interpreted in light of the implementation limitations and experimental setup.
