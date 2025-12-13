import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

# Encode target
df["Churn"] = df["Churn"].astype(str).str.strip()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop customerID
df.drop("customerID", axis=1, inplace=True)

# Encode categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # save the encoder

# Save preprocessor (LabelEncoders)
with open("preprocessor.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("Preprocessor saved as preprocessor.pkl")
