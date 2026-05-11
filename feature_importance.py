import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ---------------------------------
# PATH SETUP (SAFE)
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
data_file = BASE_DIR / "data" / "processed_dataset.csv"

print("Loading processed dataset from:", data_file)

df = pd.read_csv(data_file)

# ---------------------------------
# TARGET VARIABLE
# ---------------------------------

target = "Job_Satisfaction"

# encode target if categorical
if df[target].dtype == "object":

    le = LabelEncoder()
    df[target] = le.fit_transform(df[target])

# ---------------------------------
# FEATURE LIST
# ---------------------------------

features = [

"Expectation_Score",
"Reality_Score",
"Adaptability_Score",
"Expectation_Reality_Gap",
"Career_Growth",
"Work_Life_Balance"

]

X = df[features]
y = df[target]

# ---------------------------------
# RANDOM FOREST MODEL
# ---------------------------------

model = RandomForestClassifier(
n_estimators=400,
random_state=42
)

model.fit(X, y)

# ---------------------------------
# FEATURE IMPORTANCE
# ---------------------------------

importances = model.feature_importances_

importance_df = pd.DataFrame({

"Feature": features,
"Importance": importances

}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(importance_df)

# ---------------------------------
# VISUALIZATION
# ---------------------------------

plt.figure(figsize=(9,5))

sns.barplot(

data=importance_df,
x="Importance",
y="Feature",
hue="Feature",
palette="viridis",
legend=False

)

plt.title("Behavioral Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()

plt.show()