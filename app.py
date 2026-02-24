import streamlit as st
import numpy as np
import joblib

import os
os.chdir("C:\\Users\\sugku\\Data Science with Onur\\Project 5_Predicting Student Exam Scores with ML\\Predicting-Student-Performance-Based-on-Habits")

model = joblib.load("best_model.pkl")

st.title("Student Exam Score Predictor")

# Let include an input fields
study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 2.0) # 0.0 is min value, 12.0 is max value, & 2.0 is base value
attendance = st.slider("Attendance Percentage", 0.0, 100.0, 80.0)
mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)
part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])


# ptj = part time job
# In our model we have used part time job as 0 and 1, so lets change it to yes and no
ptj_encoded = 1 if part_time_job == "Yes" else 0


if st.button("Predict Exam Score"):

    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]]) # used encoded part time job
    prediction = model.predict(input_data)[0]

    # We need to cap between 0 to 100, becoz the score shouldn't beyong this range
    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Exam Score: {prediction: .2f}")