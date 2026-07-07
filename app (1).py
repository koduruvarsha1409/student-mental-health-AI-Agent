import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# -------------------------------
# Load Model and Encoders
# -------------------------------
with open("student_mental_health_xgboost.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.set_page_config(page_title="Student Mental Health Assessment Agent")

st.title("🧠 Student Mental Health Assessment Agent")
st.write("Enter the student details below.")

# -------------------------------
# Inputs
# -------------------------------

gender = st.selectbox(
    "Choose your gender",
    label_encoders["Choose your gender"].classes_
)

age = st.number_input(
    "Age",
    min_value=15,
    max_value=60,
    value=20
)

course = st.selectbox(
    "What is your course?",
    label_encoders["What is your course?"].classes_
)

year = st.selectbox(
    "Your current year of Study",
    label_encoders["Your current year of Study"].classes_
)

cgpa = st.selectbox(
    "What is your CGPA?",
    label_encoders["What is your CGPA?"].classes_
)

marital = st.selectbox(
    "Marital status",
    label_encoders["Marital status"].classes_
)

depression = st.selectbox(
    "Do you have Depression?",
    label_encoders["Do you have Depression?"].classes_
)

anxiety = st.selectbox(
    "Do you have Anxiety?",
    label_encoders["Do you have Anxiety?"].classes_
)

panic = st.selectbox(
    "Do you have Panic attack?",
    label_encoders["Do you have Panic attack?"].classes_
)

treatment = st.selectbox(
    "Did you seek any specialist for a treatment?",
    label_encoders["Did you seek any specialist for a treatment?"].classes_
)

# -------------------------------
# Predict
# -------------------------------

if st.button("Predict"):

    input_data = pd.DataFrame({

        "Choose your gender": [
            label_encoders["Choose your gender"].transform([gender])[0]
        ],

        "Age": [age],

        "What is your course?": [
            label_encoders["What is your course?"].transform([course])[0]
        ],

        "Your current year of Study": [
            label_encoders["Your current year of Study"].transform([year])[0]
        ],

        "What is your CGPA?": [
            label_encoders["What is your CGPA?"].transform([cgpa])[0]
        ],

        "Marital status": [
            label_encoders["Marital status"].transform([marital])[0]
        ],

        "Do you have Anxiety?": [
            label_encoders["Do you have Anxiety?"].transform([anxiety])[0]
        ],

        "Do you have Panic attack?": [
            label_encoders["Do you have Panic attack?"].transform([panic])[0]
        ],

        "Did you seek any specialist for a treatment?": [
            label_encoders["Did you seek any specialist for a treatment?"].transform([treatment])[0]
        ]

    })

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction")

    if prediction == 1:

        st.error("⚠ Student is likely to have Mental Health Issues")

        recommendation = """
### Recommendation

- Consult a mental health professional.
- Maintain a healthy sleep schedule.
- Reduce academic stress.
- Practice meditation or exercise.
- Talk with trusted family or friends.
"""

    else:

        st.success("✅ Student appears Mentally Healthy")

        recommendation = """
### Recommendation

- Continue a healthy lifestyle.
- Maintain good sleep habits.
- Exercise regularly.
- Balance study and recreation.
"""

    st.markdown(recommendation)

    # Save prediction log
    log = pd.DataFrame({
        "Date": [datetime.now()],
        "Prediction": [prediction]
    })

    if os.path.exists("prediction_logs.csv"):
        old = pd.read_csv("prediction_logs.csv")
        old = pd.concat([old, log], ignore_index=True)
        old.to_csv("prediction_logs.csv", index=False)
    else:
        log.to_csv("prediction_logs.csv", index=False)

    st.success("Prediction saved successfully.")

# -------------------------------
# History
# -------------------------------

if os.path.exists("prediction_logs.csv"):

    st.subheader("Prediction History")

    history = pd.read_csv("prediction_logs.csv")

    st.dataframe(history)

    st.download_button(
        "Download Prediction History",
        history.to_csv(index=False),
        "prediction_logs.csv",
        "text/csv"
    )
