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
    
    # Show head of the dataset
    st.subheader("📋 First 5 Rows of Uploaded Dataset")
    st.dataframe(df.head())

    # Data Preprocessing
    if all(col in df.columns for col in ["Assignment_Score", "Test_Score", "Attendance"]):
        df["Total_Score"] = (df["Assignment_Score"] + df["Test_Score"]) / 2
        X = df[["Total_Score", "Attendance"]]

        # Anomaly Detection
        model = IsolationForest(contamination=0.05, random_state=42)
        model.fit(X)
        df["anomaly"] = model.predict(X)

        # Displaying the full data
        #st.subheader("📄 Dataset Overview with Anomalies")
        #st.dataframe(df)

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

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Scatter Plot (Total Score vs Attendance)")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x="Total_Score", y="Attendance", hue="anomaly", palette={1:'blue', -1:'red'}, ax=ax)
            st.pyplot(fig)

        with col2:
            st.write("### Pie Chart of Normal vs Anomalous Students")
            pie_data = df["anomaly"].value_counts().rename(index={1:"Normal", -1:"Anomalous"})
            fig2, ax2 = plt.subplots()
            ax2.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=['skyblue', 'salmon'], startangle=140)
            ax2.axis('equal')
            st.pyplot(fig2)

        st.write("### Distribution of Total Scores (Histogram)")
        fig3, ax3 = plt.subplots()
        sns.histplot(df["Total_Score"], bins=20, kde=True, color="purple", ax=ax3)
        st.pyplot(fig3)

        st.write("### Attendance Distribution (Histogram)")
        fig4, ax4 = plt.subplots()
        sns.histplot(df["Attendance"], bins=20, kde=True, color="green", ax=ax4)
        st.pyplot(fig4)

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
