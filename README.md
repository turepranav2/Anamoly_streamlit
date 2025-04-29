
# 🎓 Anomaly Detection in Student Performance

A Streamlit-based data analytics mini project to identify underperforming or overperforming students using anomaly detection techniques on assignment scores, test scores, and attendance data.

---

## 📌 Project Overview

This project focuses on analyzing student performance data to detect anomalies that could indicate struggling or exceptionally performing students. Such analysis helps educators take timely action.

---

## 🚀 Features

- 📁 Upload your own student dataset
- 📊 Preprocess data automatically
- 🧠 Apply Isolation Forest for anomaly detection
- 📉 Display visual insights using plots
- ✅ Output includes:
  - List of anomalous students
  - Model accuracy score
  - Visual graphs for interpretation

---

## 🧪 Technologies Used

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn

---

## 📂 Dataset Format

Your CSV file must contain these columns:

```csv
Student_ID,Name,Assignment_Score,Test_Score,Attendance
```

A sample dataset of 500 students is included as `sample_data.csv`.

---

## 💻 How to Run the App Locally

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/Anamoly_streamlit.git
cd Anamoly_streamlit
```

### Step 2: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit App
```bash
streamlit run app.py
```

---

## 🌐 Live App

🔗 [Click to open the hosted Streamlit app](https://your-streamlit-url.streamlit.app)  
(Replace with your actual link)

---

## 📈 Sample Output

- 📋 Anomalous student table
- 📊 Accuracy of the model
- 📉 Visualizations like distribution plots and outlier graphs
