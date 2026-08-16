import os
import json
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (accuracy_score,roc_auc_score,precision_score,recall_score,f1_score,matthews_corrcoef,confusion_matrix,classification_report)

# Page setup
st.set_page_config(page_title="Breast Cancer Classification",layout="wide")
st.title("Breast Cancer Classification")
st.write("Machine Learning Assignment 2")

# Model files
MODEL_DIR = os.path.join(os.path.dirname(__file__),"model")
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}

# Load feature names used during training
with open(
    os.path.join(MODEL_DIR, "feature_names.json"),
    "r"
) as file:
    feature_names = json.load(file)

# Sidebar
st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose a model",
    list(MODEL_FILES.keys())
)

st.sidebar.header("Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload test CSV file",
    type=["csv"]
)

# Main application
if uploaded_file is None:
    st.info(
        "Please upload the test_data.csv file to start the evaluation."
    )
else:
    # Read uploaded CSV
    try:
        test_df = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"Unable to read the CSV file: {error}")
        st.stop()

    # Check required columns
    required_columns = feature_names + ["target"]
    missing_columns = [
        column
        for column in required_columns
        if column not in test_df.columns
    ]
    if missing_columns:
        st.error("The following required columns are missing: " + ", ".join(missing_columns))
        st.stop()

    st.subheader("Test Data")
    st.write(f"Number of test samples: {len(test_df)}")

    # Separate features and target
    X_test = test_df[feature_names]
    y_test = test_df["target"]

    # Load selected model
    model_path = os.path.join(MODEL_DIR,MODEL_FILES[selected_model])

    try:
        model = joblib.load(model_path)
    except Exception as error:
        st.error(f"Unable to load the selected model: {error}")
        st.stop()

    # Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate six required metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    # Display selected model results
    st.subheader("Selected Model")
    st.write(f"**Model:** {selected_model}")
    metric_df = pd.DataFrame(
        {
            "Metric": list(metrics.keys()),
            "Score": [
                round(value, 4)
                for value in metrics.values()
            ]
        }
    )
    st.dataframe(
        metric_df,
        use_container_width=True,
        hide_index=True
    )

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Malignant", "Actual Benign"],
        columns=["Predicted Malignant", "Predicted Benign"]
    )
    st.dataframe(
        cm_df,
        use_container_width=True
    )

    # Classification Report
    st.subheader("Classification Report")
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Malignant", "Benign"]
    )
    st.text(report)

    # Evaluate all five models
    st.subheader("Comparison of All Models")
    comparison_results = []
    for model_name, file_name in MODEL_FILES.items():
        model_path = os.path.join(
            MODEL_DIR,
            file_name
        )
        current_model = joblib.load(model_path)
        predictions = current_model.predict(X_test)
        probabilities = current_model.predict_proba(X_test)[:, 1]
        comparison_results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "AUC": roc_auc_score(y_test, probabilities),
                "Precision": precision_score(y_test, predictions),
                "Recall": recall_score(y_test, predictions),
                "F1": f1_score(y_test, predictions),
                "MCC": matthews_corrcoef(y_test, predictions)
            }
        )

    comparison_df = pd.DataFrame(
        comparison_results
    )
    # Round only for display
    metric_columns = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
    comparison_df[metric_columns] = (
        comparison_df[metric_columns].round(4)
    )
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # Best model for each metric
    st.subheader("Best Model for Each Metric")
    best_models = {}
    for metric in metric_columns:
        best_row = comparison_df.loc[
            comparison_df[metric].idxmax()
        ]
        best_models[metric] = best_row["Model"]

    best_df = pd.DataFrame(
        {
            "Metric": list(best_models.keys()),
            "Best Model": list(best_models.values())
        }
    )
    st.dataframe(
        best_df,
        use_container_width=True,
        hide_index=True
    )

    # Overall winner
    # MCC is used here as the overall comparison criterion.
    winner = comparison_df.loc[
        comparison_df["MCC"].idxmax(),
        "Model"
    ]
    
    st.success(
        f"Overall Winner based on MCC: {winner}"
    )