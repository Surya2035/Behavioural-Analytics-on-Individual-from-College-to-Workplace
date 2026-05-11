import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = BASE_DIR / "data" / "dataset.csv"
output_file = BASE_DIR / "data" / "processed_dataset.csv"

print("Loading dataset from:", input_file)

df = pd.read_csv(input_file)

expectation_cols = [
"Expect_Autonomy",
"Expect_Feedback",
"Expect_Interaction",
"Expect_Work_Value",
"Expect_High_Skill",
"Expect_Clear_Tasks"
]

reality_cols = [
"Job_Match_Expectation",
"Supportive_Workplace",
"Career_Growth",
"Work_Life_Balance"
]

adapt_cols = [
"Adapted_Quickly",
"Reality_Shock",
"Expectation_Changed"
]

df["Expectation_Score"] = df[expectation_cols].mean(axis=1)
df["Reality_Score"] = df[reality_cols].mean(axis=1)
df["Adaptability_Score"] = df[adapt_cols].mean(axis=1)

df["Expectation_Reality_Gap"] = df["Expectation_Score"] - df["Reality_Score"]

df.to_csv(output_file, index=False)

print("Processed dataset saved at:", output_file)