# 🎓 Student Performance Prediction System (ML Web App)

🚀 An end-to-end Machine Learning web application that predicts whether a student will **Pass or Fail** based on academic and behavioral features.  

Built using **Scikit-learn** and deployed using **Streamlit Cloud**.

---

## 🌐 Live Demo
👉 [https://your-streamlit-app-link.streamlit.app](https://student-performance-application.streamlit.app/)     

---

## 📌 Project Overview

This project is a **Machine Learning-based Student Performance Prediction System** designed to analyze student academic behavior and predict final outcomes.

It works like a **recommender-style scoring system**, where student input features are processed to estimate performance outcomes.

This helps in early identification of at-risk students and supports better academic intervention.

---

## 🧠 Problem Statement

Educational institutions often fail to identify struggling students early.

This system solves this by:
- Predicting student performance (Pass/Fail)
- Identifying risk factors like failures and absences
- Helping improve academic support systems

---

## ⚙️ Features Used

- 🎂 Age  
- 📚 Study Time  
- ❌ Past Failures  
- ⏰ Absences  
- 🏫 School  
- 🚻 Gender (Sex)  

---

## 🤖 Machine Learning Pipeline

1. 📥 Data Collection (UCI Student Dataset)  
2. 🧹 Data Preprocessing  
3. 🔄 One-Hot Encoding for categorical variables  
4. 📏 Feature Scaling using StandardScaler  
5. 🤖 Model Training using Logistic Regression  
6. 💾 Model Serialization using Pickle  
7. 🌐 Deployment using Streamlit Cloud  

---

## 🧪 Model Details

- Algorithm: Logistic Regression  
- Type: Binary Classification  
- Target Variable: Final Grade (G3)  
- Output:
  - ✅ Pass (1)
  - ❌ Fail (0)

---

## 📊 Dataset

- Source: UCI Machine Learning Repository  
- File: `student-mat.csv`  
- Domain: Education / Student Performance Analytics  

---

## 🏗️ Project Structure

student_performance/
│
├── app.py
├── requirements.txt
│
├── model/
│ ├── student_model.pkl
│ ├── columns.pkl
│ ├── scaler.pkl
│ ├── numeric_cols.pkl
│
├── data/
│ └── student-mat.csv
│
├── notebooks/
│ └── 01_EDA.ipynb


---

## 🚀 How to Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/student-performance-app.git
cd student-performance-app
pip install -r requirements.txt
streamlit run app.py
```

## 🌍 Deployment

• Push code to GitHub

• Connect repo in Streamlit Cloud

• Set app.py as entry point

• Deploy 🚀

## 📈 Key Highlights

• End-to-end ML pipeline

• Real-world education use case

• Streamlit interactive web app

• Feature engineering + scaling

• Production-style deployment structure

• Recommender-system style prediction logic

## 💡 Future Improvements

• Add prediction probability (confidence score)

• Try advanced ML models (Random Forest / XGBoost)

• Convert into student ranking recommender system

• Add API support for institutions

• Improve UI/UX design

## 🧠 Tech Stack

• Python 

• Pandas

• NumPy

• Scikit-learn

• Streamlit

• Pickle

## 👨‍💻 Author

Built by Shiv Mehta

Machine Learning & AI Enthusiast | Recommender Systems Explorer

## ⭐ Note

This project demonstrates a complete end-to-end ML lifecycle:

Data → Model → Deployment

It reflects real-world production-style AI system design used in recommendation and prediction systems.
