import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
data_file = BASE_DIR / "data" / "processed_dataset.csv"

df = pd.read_csv(data_file)

features = df[[
"Expectation_Score",
"Reality_Score",
"Adaptability_Score"
]]

scaler = StandardScaler()
X = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)

df["Cluster"] = kmeans.fit_predict(X)

df.to_csv(data_file, index=False)

print("Clustering completed")