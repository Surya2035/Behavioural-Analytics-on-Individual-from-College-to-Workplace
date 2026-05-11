import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------------
# CONFIG
# -----------------------------------

st.set_page_config(layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "processed_dataset.csv"
model_path = BASE_DIR / "models" / "satisfaction_model.pkl"
sem_image = BASE_DIR / "data" / "sem_model_pro.png"
path_file = BASE_DIR / "data" / "path_coefficients.csv"

# -----------------------------------
# CACHED DATA & MODEL LOADING
# -----------------------------------

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(data_path)
        
        # Ensure engineered features exist in the dataframe to prevent SHAP index errors
        if "Expectation_Reality_Gap" not in df.columns:
            df["Expectation_Reality_Gap"] = df["Expectation_Score"] - df["Reality_Score"]
        if "Gap_Strength" not in df.columns:
            df["Gap_Strength"] = df["Expectation_Reality_Gap"].abs()
        if "Adaptability_Impact" not in df.columns:
            df["Adaptability_Impact"] = df["Adaptability_Score"] * df["Reality_Score"]
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        return joblib.load(model_path)
    except Exception as e:
        return None

df = load_data()
model = load_model()

# -----------------------------------
# UI STYLE
# -----------------------------------

st.markdown("""
<style>
body {background-color: #0f172a;}
.block-container {
    background: rgba(255,255,255,0.04);
    padding: 2rem;
    border-radius: 15px;
}
h1,h2,h3 {color: #38bdf8;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 Behavioral Analytics Dashboard")

# -----------------------------------
# SIDEBAR
# -----------------------------------

section = st.sidebar.radio("Navigate", [
    "Overview",
    "3D Clustering",
    "Correlation",
    "SEM Interactive",
    "Prediction"
])

# -----------------------------------
# OVERVIEW
# -----------------------------------

if section == "Overview":

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), width="stretch")

    st.subheader("📈 Feature Averages")

    features = [
        "Expectation_Score","Reality_Score","Adaptability_Score",
        "Expectation_Reality_Gap","Career_Growth","Work_Life_Balance"
    ]

    existing_features = [f for f in features if f in df.columns]
    avg_df = df[existing_features].mean().reset_index()
    avg_df.columns = ["Feature","Average"]

    fig = px.bar(avg_df, x="Feature", y="Average", color="Feature")
    st.plotly_chart(fig, width="stretch")

# -----------------------------------
# 3D CLUSTERING
# -----------------------------------

elif section == "3D Clustering":

    fig = px.scatter_3d(
        df,
        x="Expectation_Score",
        y="Reality_Score",
        z="Adaptability_Score",
        color="Job_Satisfaction" if "Job_Satisfaction" in df.columns else None
    )
    st.plotly_chart(fig, width="stretch")

# -----------------------------------
# CORRELATION
# -----------------------------------

elif section == "Correlation":

    numeric_df = df.select_dtypes(include=['int64','float64'])
    corr = numeric_df.corr()

    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
    st.plotly_chart(fig, width="stretch")

# -----------------------------------
# SEM
# -----------------------------------

elif section == "SEM Interactive":

    st.subheader("📌 Structural Equation Model (SEM)")

    if sem_image.exists():
        st.image(str(sem_image), caption="SEM Diagram", width="stretch")
    else:
        st.warning("SEM image not found")

    st.markdown("---")

    if not path_file.exists():
        st.error("Run SEM first")
        st.stop()

    path_df = pd.read_csv(path_file)

    st.subheader("📊 Path Coefficients Table")
    st.dataframe(path_df, width="stretch")

    beta_map = dict(zip(path_df["Path"], path_df["Beta"]))

    nodes = {
        "Expectation": (1,5),
        "Reality": (1,3),
        "Adaptability": (1,1),
        "Gap": (4,3),
        "Satisfaction": (7,3),
        "Growth": (10,3)
    }

    edges = [
        ("Expectation","Gap"),
        ("Reality","Gap"),
        ("Adaptability","Satisfaction"),
        ("Gap","Satisfaction"),
        ("Satisfaction","Growth")
    ]

    fig = go.Figure()

    for name,(x,y) in nodes.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            text=name,
            marker=dict(size=45,color="#22c55e")
        ))

    for src,tgt in edges:
        x1,y1 = nodes[src]
        x2,y2 = nodes[tgt]
        beta = beta_map.get(f"{src} → {tgt}", "NA")

        fig.add_trace(go.Scatter(
            x=[x1,x2], y=[y1,y2],
            mode='lines',
            line=dict(width=4,color="#f97316")
        ))

        fig.add_annotation(x=x2,y=y2,ax=x1,ay=y1,arrowhead=3)

        fig.add_trace(go.Scatter(
            x=[(x1+x2)/2],
            y=[(y1+y2)/2],
            mode='text',
            text=[f"β={round(beta,2) if beta!='NA' else 'NA'}"]
        ))

    fig.update_layout(
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="white"),
        showlegend=False
    )

    st.plotly_chart(fig, width="stretch")

# -----------------------------------
# PREDICTION + SHAP
# -----------------------------------

elif section == "Prediction":

    st.subheader("🎯 Predict Job Satisfaction")

    if model is None:
        st.error("Model not loaded. Please check the models folder.")
        st.stop()

    # INPUTS
    exp = st.slider("Expectation Score",1,5,3)
    real = st.slider("Reality Score",1,5,3)
    adapt = st.slider("Adaptability Score",1,5,3)
    growth = st.slider("Career Growth",1,5,3)
    wlb = st.slider("Work Life Balance",1,5,3)

    # FEATURE ENGINEERING
    gap = exp - real
    gap_strength = abs(gap)
    adapt_impact = adapt * real

    input_df = pd.DataFrame({
        "Expectation_Score":[exp],
        "Reality_Score":[real],
        "Adaptability_Score":[adapt],
        "Expectation_Reality_Gap":[gap],
        "Gap_Strength":[gap_strength],
        "Adaptability_Impact":[adapt_impact],
        "Career_Growth":[growth],
        "Work_Life_Balance":[wlb]
    })

    # PREDICTION
    if st.button("Predict"):

        pred = model.predict(input_df)[0]

        # FIX: Dynamic matching based on model output type
        if isinstance(pred, str):
            # If the model outputs actual strings
            pred_lower = pred.lower()
            if "low" in pred_lower:
                st.error(f"🔴 {pred.title()}")
            elif "mod" in pred_lower or "med" in pred_lower:
                st.warning(f"🟡 {pred.title()}")
            elif "high" in pred_lower:
                st.success(f"🟢 {pred.title()}")
            else:
                st.info(f"🔵 Predicted: {pred}")

        elif isinstance(pred, (int, float, np.number)):
            pred_val = float(pred)
            
            # Check if it's a regression model (e.g. scores from 1 to 5)
            if isinstance(pred, float) or not hasattr(model, "predict_proba"):
                if pred_val >= 3.5:
                    st.success(f"🟢 High Satisfaction ({pred_val:.2f})")
                elif pred_val >= 2.5:
                    st.warning(f"🟡 Moderate Satisfaction ({pred_val:.2f})")
                else:
                    st.error(f"🔴 Low Satisfaction ({pred_val:.2f})")
                    
            # Check if it's a classifier model
            else:
                if hasattr(model, "classes_") and len(model.classes_) == 2:
                    # Binary Classification (0 = Low, 1 = High)
                    if pred_val == 0:
                        st.error("🔴 Low Satisfaction")
                    else:
                        st.success("🟢 High Satisfaction")
                else:
                    # Multi-Class Classification (0 = Low, 1 = Moderate, 2 = High)
                    if pred_val == 0:
                        st.error("🔴 Low Satisfaction")
                    elif pred_val == 1:
                        st.warning("🟡 Moderate Satisfaction")
                    elif pred_val == 2:
                        st.success("🟢 High Satisfaction")
                    else:
                        st.info(f"🔵 Predicted Class: {int(pred_val)}")

    # SHAP
    st.subheader("🧠 SHAP Explainability")

    try:
        scaler = model.named_steps["scaler"]
        rf_model = model.named_steps["rf"]

        features = [
            "Expectation_Score","Reality_Score","Adaptability_Score",
            "Expectation_Reality_Gap","Gap_Strength",
            "Adaptability_Impact","Career_Growth","Work_Life_Balance"
        ]

        X = df[features]
        X_scaled = scaler.transform(X)

        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_scaled)

        plt.clf() 
        
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X, show=False)

        st.pyplot(plt.gcf())

    except Exception as e:
        st.error(f"SHAP Error: {e}")