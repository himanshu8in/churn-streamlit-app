# Import necessary libraries
import pandas as pd
import pickle
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Load dataset
url = "https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Load preprocessor (LabelEncoders)
with open("preprocessor.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Convert TotalCharges to numeric and drop NaNs
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

# Encode target column
df["Churn"] = df["Churn"].astype(str).str.strip()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop customerID column
df.drop("customerID", axis=1, inplace=True)

# Apply saved LabelEncoders to categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    df[col] = label_encoders[col].transform(df[col])

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Initialize LightGBM model
base_model = LGBMClassifier(objective="binary", class_weight="balanced", random_state=42)

# Hyperparameter grid for RandomizedSearchCV
param_grid = {
    "num_leaves": [20, 31, 40, 60],
    "max_depth": [-1, 5, 10, 20],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [200, 400, 600],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

# Randomized Search
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=25,
    scoring="roc_auc",
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit model
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# Save trained model
with open("lgbm_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Model saved as lgbm_model.pkl")

# Evaluate model
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label="LightGBM")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Feature Importance
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(15))
plt.title("Top 15 Feature Importances")
plt.show()

# SHAP values
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
