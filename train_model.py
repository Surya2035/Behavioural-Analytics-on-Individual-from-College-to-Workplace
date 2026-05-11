import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------------
# PATH
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
data_file = BASE_DIR / "data" / "processed_dataset.csv"
model_file = BASE_DIR / "models" / "satisfaction_model.pkl"

df = pd.read_csv(data_file)

# -----------------------------------
# CREATE STRONG FEATURES
# -----------------------------------

df["Gap_Strength"] = abs(df["Expectation_Reality_Gap"])
df["Adaptability_Impact"] = df["Adaptability_Score"] * df["Reality_Score"]

# -----------------------------------
# CONVERT TO BINARY TARGET
# -----------------------------------

# 0–2 = Not satisfied
# 3–5 = Satisfied

df["Satisfaction_Binary"] = df["Job_Satisfaction"].apply(
    lambda x: 1 if x >= 3 else 0
)

target = "Satisfaction_Binary"

# -----------------------------------
# FEATURES
# -----------------------------------

features = [

"Expectation_Score",
"Reality_Score",
"Adaptability_Score",
"Expectation_Reality_Gap",
"Gap_Strength",
"Adaptability_Impact",
"Career_Growth",
"Work_Life_Balance"

]

X = df[features]
y = df[target]

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(

X, y,
test_size=0.2,
random_state=42,
stratify=y

)

# -----------------------------------
# MODEL (STRONG)
# -----------------------------------

model = Pipeline([

("scaler", StandardScaler()),

("rf", RandomForestClassifier(
    n_estimators=700,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42
))

])

model.fit(X_train, y_train)

# -----------------------------------
# PREDICTION
# -----------------------------------

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\n Model Accuracy:", round(accuracy,3))

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, pred))

# -----------------------------------
# SAVE MODEL
# -----------------------------------

joblib.dump(model, model_file)

print("\nModel saved to:", model_file)