import os
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# PROJECT FILES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}


# ============================================================
# SIMPLE CUSTOM STYLE
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #f7f8fa;
}

/* Main heading */
.main-header {
    background: linear-gradient(135deg, #172554, #164e63);
    padding: 22px 28px;
    border-radius: 12px;
    margin-bottom: 22px;
    border-left: 6px solid #2dd4bf;
}

.header-title {
    color: white;
    font-size: 30px;
    font-weight: 700;
}

.header-subtitle {
    color: #a5f3fc;
    font-size: 15px;
    margin-top: 5px;
}


/* Section headings */

.section-title {
    font-size: 19px;
    font-weight: 650;
    padding: 9px 14px;
    border-radius: 8px;
    margin-top: 24px;
    margin-bottom: 15px;
    border-left: 5px solid;
}

.section-orange {
    background-color: #fff3e0;
    color: #e65100;
    border-left-color: #fb8c00;
}

.section-blue {
    background-color: #e3f2fd;
    color: #1565c0;
    border-left-color: #42a5f5;
}

.section-green {
    background-color: #e8f5e9;
    color: #2e7d32;
    border-left-color: #66bb6a;
}

.section-purple {
    background-color: #f3e5f5;
    color: #6a1b9a;
    border-left-color: #ab47bc;
}

.section-yellow {
    background-color: #fffde7;
    color: #8d6e00;
    border-left-color: #fbc02d;
}


/* Information boxes */

.info-card {
    background-color: white;
    border: 1px solid #dbe2ea;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 15px;
}


/* Metric cards */

.metric-card {
    background-color: white;
    border: 1px solid #dbe2ea;
    border-top: 4px solid #14b8a6;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}

.metric-name {
    color: #64748b;
    font-size: 13px;
}

.metric-value {
    color: #0f172a;
    font-size: 24px;
    font-weight: 700;
    margin-top: 4px;
}


/* Winner box */

.winner-box {
    background-color: #ecfeff;
    border: 1px solid #5eead4;
    border-left: 6px solid #0d9488;
    border-radius: 10px;
    padding: 18px 22px;
    margin-top: 18px;
}

.winner-title {
    color: #0f766e;
    font-size: 13px;
    font-weight: 700;
}

.winner-model {
    color: #134e4a;
    font-size: 24px;
    font-weight: 700;
    margin-top: 4px;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #eaf2f4;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="main-header">
<div class="header-title">🩺 Breast Cancer Classification</div>
<div class="header-subtitle">Machine Learning Assignment 2 · Wisconsin Diagnostic Dataset</div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

feature_file = os.path.join(
    MODEL_DIR,
    "feature_names.json"
)

try:

    with open(feature_file, "r") as file:
        feature_names = json.load(file)

except FileNotFoundError:

    st.error(
        "feature_names.json was not found inside the model folder."
    )
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ⚙️ Controls")

st.sidebar.write(
    "Select a trained model and upload the test dataset."
)

selected_model = st.sidebar.selectbox(
    "Classification Model",
    list(MODEL_FILES.keys())
)

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
**Dataset**

Breast Cancer Wisconsin (Diagnostic)

**Models:** 5

**Evaluation metrics:** 6

Accuracy · AUC · Precision · Recall · F1 · MCC
"""
)


# ============================================================
# WAIT FOR FILE
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
<div class="info-card">
<h3 style="color:#0f766e;">Getting Started</h3>

<p>
Upload the <b>test_data.csv</b> file using the sidebar
to start the evaluation.
</p>

<p>
The application will evaluate the selected model and
compare all five saved classification models.
</p>
</div>
""",
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# READ TEST DATA
# ============================================================

try:

    test_df = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(
        f"Unable to read the uploaded CSV file: {error}"
    )
    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = feature_names + ["target"]

missing_columns = [
    column
    for column in required_columns
    if column not in test_df.columns
]

if missing_columns:

    st.error(
        "The uploaded file is missing these required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


# ============================================================
# TEST DATA OVERVIEW
# ============================================================

st.markdown(
    """
<div class="section-title section-orange">
📋 Test Data Overview
</div>
""",
    unsafe_allow_html=True
)

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "Test Samples",
        len(test_df)
    )

with info2:
    st.metric(
        "Number of Features",
        len(feature_names)
    )

with info3:
    st.metric(
        "Selected Model",
        selected_model
    )


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

X_test = test_df[feature_names]
y_test = test_df["target"]


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    MODEL_FILES[selected_model]
)

try:

    model = joblib.load(model_path)

except Exception as error:

    st.error(
        f"Unable to load the selected model: {error}"
    )
    st.stop()


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# SIX REQUIRED METRICS
# ============================================================

metrics = {
    "Accuracy": accuracy_score(
        y_test,
        y_pred
    ),

    "AUC": roc_auc_score(
        y_test,
        y_prob
    ),

    "Precision": precision_score(
        y_test,
        y_pred
    ),

    "Recall": recall_score(
        y_test,
        y_pred
    ),

    "F1": f1_score(
        y_test,
        y_pred
    ),

    "MCC": matthews_corrcoef(
        y_test,
        y_pred
    )
}


# ============================================================
# SELECTED MODEL PERFORMANCE
# ============================================================

st.markdown(
    """
<div class="section-title section-blue">
🎯 Selected Model Performance
</div>
""",
    unsafe_allow_html=True
)

st.write(
    f"Current model: **{selected_model}**"
)

metric_columns = st.columns(6)

for column, (metric_name, metric_value) in zip(
    metric_columns,
    metrics.items()
):

    with column:

        st.markdown(
            f"""
<div class="metric-card">
<div class="metric-name">{metric_name}</div>
<div class="metric-value">{metric_value:.4f}</div>
</div>
""",
            unsafe_allow_html=True
        )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.markdown(
    """
<div class="section-title section-green">
🔎 Confusion Matrix
</div>
""",
    unsafe_allow_html=True
)

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_col1, cm_col2 = st.columns([1, 1])

with cm_col1:

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="GnBu",
        cbar=False,
        xticklabels=[
            "Malignant",
            "Benign"
        ],
        yticklabels=[
            "Malignant",
            "Benign"
        ],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(
        f"Confusion Matrix - {selected_model}"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


with cm_col2:

    st.markdown(
        """
<div class="info-card">

<h4 style="color:#2e7d32;">
Understanding the Matrix
</h4>

<p>
The diagonal cells represent correctly classified
observations.
</p>

<p>
The off-diagonal cells represent incorrect predictions.
</p>

<p>
For this medical classification problem, a malignant
case predicted as benign is a false negative and is
particularly important.
</p>

</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.markdown(
    """
<div class="section-title section-purple">
📑 Classification Report
</div>
""",
    unsafe_allow_html=True
)

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Malignant",
        "Benign"
    ],
    output_dict=True
)

report_df = (
    pd.DataFrame(report)
    .transpose()
    .round(4)
)

st.dataframe(
    report_df,
    use_container_width=True
)


# ============================================================
# COMPARE ALL FIVE MODELS
# ============================================================

st.markdown(
    """
<div class="section-title section-yellow">
📊 Model Performance Comparison
</div>
""",
    unsafe_allow_html=True
)

comparison_results = []

for model_name, file_name in MODEL_FILES.items():

    current_model_path = os.path.join(
        MODEL_DIR,
        file_name
    )

    try:

        current_model = joblib.load(
            current_model_path
        )

        predictions = current_model.predict(
            X_test
        )

        probabilities = current_model.predict_proba(
            X_test
        )[:, 1]

        comparison_results.append(
            {
                "Model": model_name,

                "Accuracy": accuracy_score(
                    y_test,
                    predictions
                ),

                "AUC": roc_auc_score(
                    y_test,
                    probabilities
                ),

                "Precision": precision_score(
                    y_test,
                    predictions
                ),

                "Recall": recall_score(
                    y_test,
                    predictions
                ),

                "F1": f1_score(
                    y_test,
                    predictions
                ),

                "MCC": matthews_corrcoef(
                    y_test,
                    predictions
                )
            }
        )

    except Exception as error:

        st.warning(
            f"Could not evaluate {model_name}: {error}"
        )


comparison_df = pd.DataFrame(
    comparison_results
)


metric_columns = [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC"
]


# Display rounded values without changing
# the actual values used for comparison.

display_df = comparison_df.copy()

display_df[metric_columns] = (
    display_df[metric_columns].round(4)
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# BEST MODEL FOR EACH METRIC
# ============================================================

st.markdown(
    """
<div class="section-title section-green">
🏅 Best Model by Metric
</div>
""",
    unsafe_allow_html=True
)

best_models = {}

for metric in metric_columns:

    best_index = comparison_df[
        metric
    ].idxmax()

    best_models[metric] = comparison_df.loc[
        best_index,
        "Model"
    ]


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


# ============================================================
# OVERALL WINNER
# ============================================================

winner_index = comparison_df[
    "MCC"
].idxmax()

winner = comparison_df.loc[
    winner_index,
    "Model"
]

winner_mcc = comparison_df.loc[
    winner_index,
    "MCC"
]


st.markdown(
    f"""
<div class="winner-box">

<div class="winner-title">
★ OVERALL WINNER
</div>

<div class="winner-model">
{winner}
</div>

<p>
The overall winner is selected using the highest
Matthews Correlation Coefficient (MCC).
</p>

<p>
<b>MCC = {winner_mcc:.4f}</b>
</p>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# CONCLUSION
# ============================================================

st.markdown(
    """
<div class="section-title section-orange">
📝 Conclusion
</div>
""",
    unsafe_allow_html=True
)

st.markdown(
    f"""
<div class="info-card">

<p>
Based on the uploaded test dataset, 
<b>{winner}</b> achieved the highest MCC among
the five classification models.
</p>

<p>
The models were evaluated using six metrics:
<b>Accuracy, AUC, Precision, Recall, F1-score and MCC.</b>
</p>

<p>
All the values displayed above are calculated dynamically
from the predictions generated by the saved models.
</p>

</div>
""",
    unsafe_allow_html=True
)