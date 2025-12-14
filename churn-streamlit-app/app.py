import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page configuration
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
    "Predict customer churn using a LightGBM model "
    "and explain individual predictions using SHAP."
)

# -------------------------------------------------
# Load template dataset (for UI consistency)
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
# Sidebar input
# -------------------------------------------------
st.sidebar.header("Customer Information")

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
# Prediction and SHAP explanation
# -------------------------------------------------
if st.button("Predict Churn"):

    # Prediction
    churn_proba = model.predict_proba(input_df)[:, 1][0]
    churn_pred = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.metric("Churn Probability", f"{churn_proba * 100:.2f}%")

    if churn_pred == 1:
        st.error("Prediction: Customer is likely to churn")
    else:
        st.success("Prediction: Customer is not likely to churn")

    # -------------------------------------------------
    # SHAP Explanation (Version-safe)
    # -------------------------------------------------
    st.subheader("Feature Importance (SHAP values)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

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

    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[1]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns,
        ),
        show=False
    )
    st.pyplot(fig2)
