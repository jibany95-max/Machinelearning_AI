import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler

model = joblib.load('classification_model.pkl')
scaler = joblib.load('scaler.pkl')  

st.title("Student Performance Prediction")

# Input fields for user to enter data
hours_studied = st.number_input("Hours Studied")
attendance = st.number_input("Attendance")

# Predict button
if st.button("Predict"):
    # Scale the input data
    input_data = np.array([[hours_studied, attendance]])
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    
    # Display result
    if prediction[0] == 1:
        st.success("The student is likely to pass.")
    else:
        st.error("The student is likely to fail.")
