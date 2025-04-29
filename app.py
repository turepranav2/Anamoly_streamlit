
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Anomaly Detection in Student Performance", layout="wide")

st.title("📊 Anomaly Detection in Student Performance")
st.write("Upload a student dataset to detect performance anomalies based on attendance, assignments, and test scores.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Dataset")
    st.dataframe(df.head())

    # Preprocessing
    if 'Total_Score' not in df.columns:
        df["Total_Score"] = (df["Assignment_Score"] + df["Test_Score"]) / 2

    # Visualization
    st.subheader("📈 Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Score Distribution")
        fig, ax = plt.subplots()
        df[["Assignment_Score", "Test_Score", "Total_Score"]].plot(kind='box', ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("Attendance vs Total Score")
        fig, ax = plt.subplots()
        sns.scatterplot(x="Attendance", y="Total_Score", data=df, ax=ax)
        st.pyplot(fig)

    # Feature Scaling
    features = ["Attendance", "Assignment_Score", "Test_Score", "Total_Score"]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df["Anomaly"] = model.fit_predict(X_scaled)
    anomalies = df[df["Anomaly"] == -1]

    st.subheader("🚨 Anomalous Students Detected")
    st.write(anomalies[["Student_ID", "Attendance", "Assignment_Score", "Test_Score", "Total_Score"]])

    st.success(f"✅ Total Students: {len(df)} | Anomalies Detected: {len(anomalies)}")
    
    st.subheader("📤 Download Anomaly Report")
    csv = anomalies.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="anomalies.csv", mime="text/csv")
