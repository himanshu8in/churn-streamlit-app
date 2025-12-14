import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# -------------------------------------------------
# Load model and preprocessor
# -------------------------------------------------
@st.cache_data
def load_artifacts():
    with open("churn-streamlit-app/model/lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("churn-streamlit-app/model/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


model, preprocessor = load_artifacts()

# -------------------------------------------------
# App UI
# -------------------------------------------------
st.title("Customer Churn Prediction")
st.write(
    "Predict whether a customer will churn using a trained LightGBM model "
    "and understand the decision using SHAP explainability."
)

# -------------------------------------------------
# Load template data (for UI ranges & categories)
# -------------------------------------------------
@st.cache_data
def load_template_data():
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


template_df = load_template_data()
feature_cols = [c for c in template_df.columns if c != "Churn"]

# -------------------------------------------------
# User Inputs
# -------------------------------------------------
st.sidebar.header("Customer Features")

user_input = {}

for col in feature_cols:
    if template_df[col].dtype == "object":
        user_input[col] = st.sidebar.selectbox(
            col,
            sorted(template_df[col].unique())
        )
    else:
        user_input[col] = st.sidebar.number_input(
            col,
            min_value=float(template_df[col].min()),
            max_value=float(template_df[col].max()),
            value=float(template_df[col].mean()),
        )

input_df = pd.DataFrame([user_input])

# -------------------------------------------------
# Apply preprocessing (LabelEncoders)
# -------------------------------------------------
categorical_cols = input_df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    input_df[col] = preprocessor[col].transform(input_df[col])

# -------------------------------------------------
# Prediction + SHAP
# -------------------------------------------------
if st.button("Predict Churn"):

    pred_proba = model.predict_proba(input_df)[:, 1][0]
    pred_class = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{pred_proba * 100:.2f}%")

    if pred_class == 1:
        st.error("Prediction: Customer is likely to churn")
    else:
        st.success("Prediction: Customer is not likely to churn")

    # -------------------------------------------------
    # SHAP Explainability (FIXED)
    # -------------------------------------------------
    st.subheader("Feature Importance (SHAP values)")

    explainer = shap.TreeExplainer(model)

    # For binary classification → use class 1 (churn)
    shap_values = explainer.shap_values(input_df)[1]

    fig, ax = plt.subplots(figsize=(8, 4))
    shap.summary_plot(
        shap_values,
        input_df,
        plot_type="bar",
        show=False
    )
    st.pyplot(fig)

    # -------------------------------------------------
    # SHAP Waterfall (Single Prediction)
    # -------------------------------------------------
    st.subheader("SHAP Waterfall (Single Prediction)")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value[1],
            data=input_df.iloc[0],
            feature_names=input_df.columns,
        ),
        show=False
    )
    st.pyplot(fig2)
