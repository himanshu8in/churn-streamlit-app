# churn-streamlit-app/app.py

import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# -----------------------
# Load model and preprocessor
# -----------------------
@st.cache_data
def load_model():
    with open("churn-streamlit-app//model/lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("churn-streamlit-app/model/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


model, preprocessor = load_model()

# -----------------------
# App UI
# -----------------------
st.title("Customer Churn Prediction")
st.write("Predict whether a customer will churn using a trained LightGBM model.")

# -----------------------
# Load template data
# -----------------------
@st.cache_data
def load_template():
    url = (
        "https://raw.githubusercontent.com/"
        "alexeygrigorev/mlbookcamp-code/master/"
        "chapter-03-churn-prediction/"
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )
    df = pd.read_csv(url)
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    return df


template_df = load_template()
feature_cols = [c for c in template_df.columns if c != "Churn"]

# -----------------------
# User Inputs
# -----------------------
st.sidebar.header("Customer Features")

user_input = {}
for col in feature_cols:
    if template_df[col].dtype == "object":
        user_input[col] = st.sidebar.selectbox(
            col, sorted(template_df[col].unique())
        )
    else:
        user_input[col] = st.sidebar.number_input(
            col,
            min_value=float(template_df[col].min()),
            max_value=float(template_df[col].max()),
            value=float(template_df[col].mean()),
        )

input_df = pd.DataFrame([user_input])

# -----------------------
# Apply Preprocessing
# -----------------------
categorical_cols = input_df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    input_df[col] = preprocessor[col].transform(input_df[col])

# -----------------------
# Prediction + SHAP
# -----------------------
if st.button("Predict Churn"):

    pred_proba = model.predict_proba(input_df)[:, 1][0]
    pred_class = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {pred_proba * 100:.2f}%")
    st.write(
        f"**Prediction:** {'Churn' if pred_class == 1 else 'No Churn'}"
    )

    # -----------------------
    # SHAP Explanation
    # -----------------------
    st.subheader("Feature Importance (SHAP values)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    fig = plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_values,
        input_df,
        matplotlib=True,
    )
    st.pyplot(fig, bbox_inches="tight")
