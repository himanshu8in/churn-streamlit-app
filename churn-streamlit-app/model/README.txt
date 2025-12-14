Customer Churn Prediction - Streamlit App

Project Structure:
churn-streamlit-app/
│
├── app.py                 # Streamlit app code (UI + prediction + SHAP)
├── requirements.txt       # List of required Python libraries
├── README.md              # GitHub formatted README
└── model/
    ├── lgbm_model.pkl     # Trained LightGBM model
    └── preprocessor.pkl   # LabelEncoders for categorical preprocessing

Setup Instructions (Local):

1. Clone the repository:
   git clone https://github.com/your-username/churn-streamlit-app.git
   cd churn-streamlit-app

2. Create a virtual environment (optional but recommended):
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows

3. Install required libraries:
   pip install -r requirements.txt

4. Run the Streamlit app:
   streamlit run app.py

5. Open the URL shown in the terminal (usually http://localhost:8501) to use the app.

Notes:
- Make sure 'model/lgbm_model.pkl' and 'model/preprocessor.pkl' exist.
- If you add new libraries, update requirements.txt:
   pip freeze > requirements.txt
