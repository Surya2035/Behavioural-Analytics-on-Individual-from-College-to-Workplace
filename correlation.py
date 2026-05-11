import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------
# PATH SETUP (FIXED - no errors)
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
data_file = BASE_DIR / "data" / "processed_dataset.csv"
output_file = BASE_DIR / "data" / "correlation_heatmap.png"

# -----------------------------------
# LOAD DATA
# -----------------------------------

print("Loading dataset from:", data_file)

df = pd.read_csv(data_file)

# -----------------------------------
# SELECT NUMERIC COLUMNS ONLY
# -----------------------------------

numeric_df = df.select_dtypes(include=['int64', 'float64'])

# -----------------------------------
# CORRELATION MATRIX
# -----------------------------------

corr = numeric_df.corr()

# -----------------------------------
# PLOT HEATMAP WITH VALUES
# -----------------------------------

plt.figure(figsize=(12, 10))

sns.heatmap(
    corr,
    annot=True,              # ✅ SHOW VALUES
    fmt=".2f",               # ✅ 2 decimal places
    cmap="coolwarm",
    linewidths=0.5,
    linecolor='black',
    square=True,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 9, "weight": "bold"}  # ✅ Better readability
)

# -----------------------------------
# TITLE
# -----------------------------------

plt.title(
    "Correlation Heatmap (Behavioral Analytics)",
    fontsize=14,
    fontweight="bold"
)

# Rotate labels for clarity
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()

# -----------------------------------
# SAVE OUTPUT
# -----------------------------------

plt.savefig(output_file, dpi=300)

print("Correlation heatmap saved at:", output_file)

plt.show()