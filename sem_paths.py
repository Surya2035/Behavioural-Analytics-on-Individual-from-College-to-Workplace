import pandas as pd
import statsmodels.api as sm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "processed_dataset.csv"
path_output = BASE_DIR / "data" / "path_coefficients.csv"
r2_output = BASE_DIR / "data" / "r2_scores.csv"

df = pd.read_csv(data_path)

# -----------------------------------
# MODEL 1: Satisfaction
# -----------------------------------

X1 = df[[
"Expectation_Score",
"Reality_Score",
"Adaptability_Score",
"Expectation_Reality_Gap"
]]

y1 = df["Job_Satisfaction"]

X1 = sm.add_constant(X1)

model1 = sm.OLS(y1, X1).fit()

# -----------------------------------
# MODEL 2: Long-Term Intention
# -----------------------------------

X2 = df[[
"Job_Satisfaction",
"Adaptability_Score"
]]

y2 = df["Career_Growth"]

X2 = sm.add_constant(X2)

model2 = sm.OLS(y2, X2).fit()

# -----------------------------------
# PATH COEFFICIENTS
# -----------------------------------

path_df = pd.DataFrame({

"Path":[
"Expectation → Satisfaction",
"Reality → Satisfaction",
"Adaptability → Satisfaction",
"Gap → Satisfaction",
"Satisfaction → Growth",
"Adaptability → Growth"
],

"Beta":[
model1.params.iloc[1],
model1.params.iloc[2],
model1.params.iloc[3],
model1.params.iloc[4],
model2.params.iloc[1],
model2.params.iloc[2]
],

"p_value":[
model1.pvalues.iloc[1],
model1.pvalues.iloc[2],
model1.pvalues.iloc[3],
model1.pvalues.iloc[4],
model2.pvalues.iloc[1],
model2.pvalues.iloc[2]
]

})

path_df.to_csv(path_output, index=False)

# -----------------------------------
# R2 VALUES
# -----------------------------------

r2_df = pd.DataFrame({
"Construct":["Satisfaction","Growth"],
"R2":[model1.rsquared, model2.rsquared]
})

r2_df.to_csv(r2_output, index=False)

print("✅ SEM paths + R2 saved")