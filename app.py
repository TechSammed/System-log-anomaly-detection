import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="System Log Anomaly Detection",
    layout="wide"
)

st.title("🚨 System Log Anomaly Detection Dashboard")
st.write("Unsupervised anomaly detection on HDFS system logs")

# -----------------------------
# Load data & model
# -----------------------------
@st.cache_data
def load_features():
    return pd.read_csv("features.csv")

@st.cache_resource
def load_model():
    return joblib.load("isolation_forest.pkl")

features = load_features()
model = load_model()

# -----------------------------
# Prepare data
# -----------------------------
X = features.drop(columns=["anomaly", "score"], errors="ignore")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Predict (in case not present)
features["anomaly"] = model.predict(X_scaled)
features["score"] = model.decision_function(X_scaled)

# -----------------------------
# Sidebar stats
# -----------------------------
st.sidebar.header("📊 Summary")

total_windows = len(features)
anomalies = (features["anomaly"] == -1).sum()

st.sidebar.metric("Total Windows", total_windows)
st.sidebar.metric("Anomalous Windows", anomalies)

# -----------------------------
# Anomaly score over time
# -----------------------------
st.subheader("📉 Anomaly Score over Time")

fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(features.index, features["score"], label="Anomaly Score")

threshold = features["score"].quantile(0.03)
ax1.axhline(threshold, color="red", linestyle="--", label="Threshold")

ax1.set_xlabel("Window Index")
ax1.set_ylabel("Score")
ax1.legend()

st.pyplot(fig1)

# -----------------------------
# PCA visualization
# -----------------------------
st.subheader("🧭 PCA Visualization of Log Windows")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig2, ax2 = plt.subplots(figsize=(8, 6))
scatter = ax2.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=features["anomaly"],
    cmap="coolwarm",
    s=15
)

ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax2.set_title("Normal vs Anomalous Windows")

st.pyplot(fig2)

# -----------------------------
# Anomaly table
# -----------------------------
st.subheader("🚨 Detected Anomalies")

anomaly_df = features[features["anomaly"] == -1].copy()
anomaly_df = anomaly_df.sort_values("score")

st.dataframe(anomaly_df.head(50))
