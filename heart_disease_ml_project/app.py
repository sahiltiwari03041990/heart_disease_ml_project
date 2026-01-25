import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("🫀 Heart Disease Prediction – ML Models")

st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox(
    "Select Model",
    ["logistic_regression","decision_tree","knn","naive_bayes","random_forest","xgboost"]
)

uploaded_file = st.file_uploader("Upload CSV test data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'target' in data.columns:
        X = data.drop('target', axis=1)
        y = data['target']
    else:
        X = data
        y = None

    model = joblib.load(f"models/{model_name}.pkl")
    preds = model.predict(X)

    st.subheader("Predictions")
    st.write(pd.DataFrame(preds, columns=["Prediction"]))

    if y is not None:
        st.subheader("Evaluation Metrics")
        report = classification_report(y, preds, output_dict=True)
        st.dataframe(pd.DataFrame(report).T)

        cm = confusion_matrix(y, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
