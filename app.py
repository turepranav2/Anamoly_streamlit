
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Anomaly Detection in Student Performance", layout="wide")

st.title("🎯 Anomaly Detection in Student Performance")

uploaded_file = st.file_uploader("Upload the student dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Data Preprocessing
    if all(col in df.columns for col in ["Assignment_Score", "Test_Score", "Attendance"]):
        df["Total_Score"] = (df["Assignment_Score"] + df["Test_Score"]) / 2
        X = df[["Total_Score", "Attendance"]]

        # Anomaly Detection
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X)
        df["anomaly"] = model.predict(X)

        # Displaying the data
        st.subheader("Dataset Overview")
        st.dataframe(df)

        # Showing anomalies
        anomalies = df[df["anomaly"] == -1]
        st.subheader("⚠️ Anomalous Students")
        st.dataframe(anomalies)

        # Accuracy calculation
        normal = df[df["anomaly"] == 1]
        accuracy = len(normal) / len(df)
        st.success(f"Model Accuracy (approx.): {accuracy*100:.2f}%")

        # Visualizations
        st.subheader("📊 Visualizations")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="Total_Score", y="Attendance", hue="anomaly", palette={1:'blue', -1:'red'}, ax=ax)
        st.pyplot(fig)

        # User Input for checking single student
        st.header("🔍 Check Single Student Performance")

        with st.form(key='student_form'):
            student_id = st.text_input("Student ID")
            name = st.text_input("Name")
            assignment_score = st.number_input("Assignment Score", min_value=0, max_value=100)
            test_score = st.number_input("Test Score", min_value=0, max_value=100)
            attendance = st.number_input("Attendance (%)", min_value=0, max_value=100)
            submit_button = st.form_submit_button(label='Check Anomaly')

        if submit_button:
            total_score = (assignment_score + test_score) / 2
            input_features = np.array([[total_score, attendance]])
            prediction = model.predict(input_features)

            st.subheader(f"Result for {name or student_id}:")
            if prediction[0] == -1:
                st.error("⚠️ The student is Anomalous (underperforming or overperforming).")
            else:
                st.success("✅ The student is Normal.")
    else:
        st.error("Uploaded CSV must have 'Assignment_Score', 'Test_Score', and 'Attendance' columns.")
else:
    st.info("Please upload a CSV file to proceed.")