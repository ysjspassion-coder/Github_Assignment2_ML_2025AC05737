# ML Assignment 2

**Course:** Machine Learning (M.Tech AIML/DSE)

---

## a. Problem Statement

The objective of this assignment is to build and compare classification models for predicting whether a breast tumor is **Malignant (0)** or **Benign (1)**.

The Breast Cancer Wisconsin (Diagnostic) dataset is used for this binary classification problem. Five classification models(logistic regression, random forest, naive bayes, knn, decision tree) are trained on the same dataset and evaluated using six required performance metrics: Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The trained models are also made available through a Streamlit application so that the models can be evaluated on uploaded test data.

---

## b. Dataset Description

| Property | Details |
|----------|---------|
| Dataset | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository |
| Number of samples | 569 |
| Number of features | 30 |
| Problem type | Binary Classification |
| Target | 0 = Malignant, 1 = Benign |

The dataset is loaded using `sklearn.datasets.load_breast_cancer()`, which provides the Breast Cancer Wisconsin (Diagnostic) dataset originally obtained from the UCI Machine Learning Repository.

An **80:20 stratified train-test split** is used with `random_state = 27`.

- Training samples: **455**
- Test samples: **114**
- Stratification is applied using the target column so that the class distribution is maintained in both sets.

The test portion is saved as `test_data.csv` for use by the Streamlit application.

---

## c. GitHub Repository Link

**GitHub Repository:** https://github.com/ysjspassion-coder/Github_Assignment2_ML_2025AC05737

The repository contains the required notebook, Streamlit application, requirements file, test data, README, and saved model files.

---

## d. Models Used

The following five classification models are implemented using the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

Standardization is used with Logistic Regression and kNN through pipelines. Random Forest is configured with 150 trees.

### Comparison Table

The following results are obtained on the held-out test data:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9825 | 0.9977 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9298 | 0.9345 | 0.9706 | 0.9167 | 0.9429 | 0.8545 |
| kNN | 0.9737 | 0.9854 | 0.9726 | 0.9861 | 0.9793 | 0.9433 |
| Naive Bayes | 0.9737 | 0.9894 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Random Forest | 0.9474 | 0.9769 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieved the highest Accuracy, AUC, Precision, F1 Score and MCC among the five models. The standardized features work well with this linear classifier on this dataset. |
| **Decision Tree** | Produced the lowest Accuracy, AUC, Recall, F1 Score and MCC among the models. Although the model is easy to interpret, its performance was lower on this test split. |
| **kNN** | Performed strongly with high Accuracy and Recall. Feature standardization is important because kNN relies on distances between observations. |
| **Naive Bayes** | Achieved the highest Recall of 1.0000, meaning it detected all positive-class samples in this test set. Its overall Accuracy and Precision were lower than Logistic Regression. |
| **Random Forest** | Performed better than the single Decision Tree on most metrics, but its performance was below Logistic Regression on this test set. |
| **Overall Winner** | **Logistic Regression**, selected using the highest Matthews Correlation Coefficient (MCC = 0.9623) — the same criterion used by the Streamlit app to pick the winner. It also independently won 5 of the 6 individual metrics, reinforcing the result. Naive Bayes achieved the highest Recall. |

### Metric-wise Best Models

| Metric | Best Model |
|--------|------------|
| Accuracy | Logistic Regression |
| AUC | Logistic Regression |
| Precision | Logistic Regression |
| Recall | Naive Bayes |
| F1 Score | Logistic Regression |
| MCC | Logistic Regression |

Therefore, **Logistic Regression is selected as the overall best-performing model based on the highest MCC**, while **Naive Bayes has the best Recall** on the test data.

**Why MCC:** MCC is used as the primary criterion because it accounts for all four confusion-matrix outcomes and stays reliable even under the dataset's mild class imbalance (357 Benign vs 212 Malignant), unlike Accuracy or F1 which can be skewed by the majority class.

---

## Streamlit Application

The Streamlit application provides the following required features:

- Upload test data in CSV format
- Select one of the five trained models
- Display the six evaluation metrics
- Display the confusion matrix
- Display the classification report
- Compare the results of all five models on the uploaded test data
- Dynamically identify the best model for each metric
- Display the overall winner based on the highest MCC (Matthews Correlation Coefficient)

Only the test data is uploaded to the Streamlit application; the trained models are loaded from the `model/` folder.

### Project Structure

```text
project-folder/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── 2025AC05737_Assignment2_ML.ipynb
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── feature_names.json
    └── metrics.json
```

### How to Run Locally

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Upload `test_data.csv` through the application and select the required model from the sidebar.

### Streamlit Community Cloud

The application can be deployed using Streamlit Community Cloud by connecting the GitHub repository and selecting `app.py` as the application file.

**Live App Link:** https://appassignment2ml2025ac05737-vrpy23mhbvw6cjxozu4zu6.streamlit.app/

---

## Requirements

The project uses the following Python packages:

- Streamlit
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- joblib

The exact dependencies are listed in `requirements.txt`.

---

## Assignment Submission Notes

The notebook contains the dataset loading, train-test split, model training, evaluation of all six required metrics, model comparison, confusion matrix, and model saving steps.

The assignment was performed in the required BITS Virtual Lab environment, and the required screenshot is included with the final submission pdf.
