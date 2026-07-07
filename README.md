# 🧠 Student Mental Health Assessment Agent

## 📌 Overview

The **Student Mental Health Assessment Agent** is an AI-powered web application that predicts whether a student is at risk of mental health issues based on survey responses. The application uses a machine learning model trained on the Student Mental Health dataset and provides personalized wellness recommendations.

This project was developed as part of the **AI & Machine Learning Phase 2 Project**, where the objective is to build an intelligent agent capable of analyzing student survey data, predicting mental health conditions, generating recommendations, and maintaining student wellness records.

---

## 🚀 Features

* Predicts student mental health status using Machine Learning.
* User-friendly Streamlit web interface.
* Data preprocessing using Label Encoding and Standard Scaling.
* Personalized wellness recommendations based on prediction.
* Stores prediction history automatically.
* Download prediction history as a CSV file.
* Fast and interactive prediction system.

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* XGBoost / Random Forest
* Joblib
* Pickle

---

## 📂 Project Structure

```text
Student_Mental_Health_Agent/
│
├── app.py
├── best_model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── prediction_logs.csv
├── requirements.txt
├── README.md
└── Student Mental Health.csv
```

---

## 📊 Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Label Encoding
4. Feature Scaling
5. Model Training
6. Model Evaluation
7. Model Selection
8. Model Deployment
9. Prediction Generation
10. Recommendation Generation

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/student-mental-health-agent.git
```

Navigate to the project folder:

```bash
cd student-mental-health-agent
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

---

## 📈 Model Evaluation Metrics

The machine learning models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Cross Validation

The best-performing model was selected and deployed in the application.

---

## 📥 Input Features

The application accepts student information such as:

* Gender
* Age
* Course
* Year of Study
* CGPA
* Marital Status
* Depression Status
* Anxiety Status
* Panic Attack History
* Treatment History

---

## 📤 Output

The application predicts the student's mental health status and provides:

* Mental Health Prediction
* Personalized Wellness Recommendations
* Prediction History Logging

---

## 📋 Future Enhancements

* PDF report generation.
* Email notifications.
* Counselor appointment integration.
* Dashboard for analytics and visualization.
* Authentication for students and administrators.
* Cloud deployment.

---

## 📚 Dataset

**Student Mental Health Dataset**

https://www.kaggle.com/datasets/shariful07/student-mental-health

---

## 👨‍💻 Author

**Koduru Varsha**

Machine Learning & AI Enthusiast

---

## ⭐ Acknowledgements

* Kaggle for providing the Student Mental Health Dataset.
* Streamlit for the web application framework.
* Scikit-learn for machine learning tools.
* Open-source Python community for supporting libraries.

---

## 📄 License

This project is developed for educational and learning purposes.
