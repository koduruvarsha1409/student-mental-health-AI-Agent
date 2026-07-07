import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# ----------------------------
# Load Model and Files
# ----------------------------

with open("student_mental_health_xgboost.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Student Mental Health Assessment Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Student Mental Health Assessment Agent")
st.write("Predict student mental health status and provide wellness recommendations.")

st.markdown("---")

# ----------------------------
# Student Details
# ----------------------------

name = st.text_input("Student Name")

gender = st.selectbox(
    "Gender",
    label_encoders["Gender"].classes_
)

age = st.number_input(
    "Age",
    min_value=15,
    max_value=60,
    value=20
)

course = st.selectbox(
    "Course",
    label_encoders["Course"].classes_
)

year = st.selectbox(
    "Year of Study",
    label_encoders["Year"].classes_
)

cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0
)

married = st.selectbox(
    "Marital Status",
    label_encoders["Married"].classes_
)

depression = st.selectbox(
    "Depression",
    label_encoders["Depression"].classes_
)

anxiety = st.selectbox(
    "Anxiety",
    label_encoders["Anxiety"].classes_
)

panic = st.selectbox(
    "Panic Attack",
    label_encoders["Panic"].classes_
)

treatment = st.selectbox(
    "Treatment",
    label_encoders["Treatment"].classes_
)

# ----------------------------
# Prediction Button
# ----------------------------

if st.button("Predict Mental Health"):

    gender = label_encoders["Gender"].transform([gender])[0]
    course = label_encoders["Course"].transform([course])[0]
    year = label_encoders["Year"].transform([year])[0]
    married = label_encoders["Married"].transform([married])[0]
    depression = label_encoders["Depression"].transform([depression])[0]
    anxiety = label_encoders["Anxiety"].transform([anxiety])[0]
    panic = label_encoders["Panic"].transform([panic])[0]
    treatment = label_encoders["Treatment"].transform([treatment])[0]

    input_data = np.array([[
        gender,
        age,
        course,
        year,
        cgpa,
        married,
        depression,
        anxiety,
        panic,
        treatment
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.markdown("---")

    if prediction == 1:

        st.error("⚠ Student is At Risk")

        recommendation = """
✔ Consult the college counselor.

✔ Maintain a proper sleep schedule.

✔ Practice meditation.

✔ Reduce academic stress.

✔ Talk to trusted family members.

✔ Seek professional psychological support if symptoms persist.
"""

    else:

        st.success("✅ Student is Mentally Healthy")

        recommendation = """
✔ Continue healthy lifestyle.

✔ Exercise regularly.

✔ Sleep 7-8 hours.

✔ Take study breaks.

✔ Stay socially connected.
"""

    st.subheader("Recommendation")
    st.write(recommendation)

    # ----------------------------
    # Save Prediction Log
    # ----------------------------

    record = pd.DataFrame({

        "Date":[datetime.now()],
        "Student":[name],
        "Prediction":[prediction],
        "Recommendation":[recommendation]

    })

    file = "prediction_logs.csv"

    if os.path.exists(file):

        old = pd.read_csv(file)

        old = pd.concat([old,record],ignore_index=True)

        old.to_csv(file,index=False)

    else:

        record.to_csv(file,index=False)

    st.success("Prediction saved successfully.")

# ----------------------------
# View Prediction History
# ----------------------------

st.markdown("---")

st.subheader("Prediction History")

if os.path.exists("prediction_logs.csv"):

    history = pd.read_csv("prediction_logs.csv")

    st.dataframe(history)

    csv = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Prediction History",
        csv,
        "prediction_logs.csv",
        "text/csv"
    )

else:

    st.info("No prediction history available.")
