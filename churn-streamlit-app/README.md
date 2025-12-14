# Customer Churn Prediction - Streamlit App

Predict whether a customer will churn using a trained LightGBM model. The app provides probability, predicted class, and SHAP feature explanations.

---

## Project Structure

churn-streamlit-app/
│
├── app.py                 # Streamlit application (UI + prediction + SHAP)
├── requirements.txt       # All required libraries
├── README.md              # How to run & deploy instructions
│
└── model/
    ├── README.txt         # Instructions for model files
    ├── lgbm_model.pkl     # (you will add this)
    └── preprocessor.pkl  # (you will add this)
