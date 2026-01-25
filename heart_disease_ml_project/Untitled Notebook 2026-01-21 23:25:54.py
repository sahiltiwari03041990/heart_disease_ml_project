# Databricks notebook source
# MAGIC %pip install xgboost joblib seaborn streamlit

# COMMAND ----------


from src.data_preprocessing import load_and_preprocess


X_train, X_test, y_train, y_test = load_and_preprocess(
    "/Volumes/nonprod_u2prepay/prepay_common/playground/Tsahil/heart.csv"
)


# COMMAND ----------

# MAGIC %sh 
# MAGIC cat /Volumes/nonprod_u2prepay/prepay_common/playground/Tsahil/heart.csv

# COMMAND ----------


from src.train_models import train_all_models

models = train_all_models(X_train, y_train)


# COMMAND ----------


from src.evaluate_models import evaluate

results_df = evaluate(models, X_test, y_test)
display(results_df)


# COMMAND ----------


import streamlit as st
print("Streamlit app is ready for deployment")
