import streamlit as st
import pandas as pd
import sys
import os

# Add the directory containing predict_salary.py to the Python path
# This assumes predict_salary.py is in the same directory as app.py
sys.path.append(os.path.dirname(__file__))

# Import the predict_salary function
try:
    from predict_salary import predict_salary
except ModuleNotFoundError:
    st.error("Error: predict_salary.py not found. Make sure it's in the same directory.")
    st.stop()

st.set_page_config(page_title="Salary Predictor App")
st.title("💰 Salary Predictor App")
st.write("Enter the details below to predict the salary.")

# Input fields for the user
age = st.slider("Age", 18, 65, 30)
years_of_experience = st.slider("Years of Experience", 0.0, 40.0, 5.0, step=0.5)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
education_level = st.selectbox("Education Level", [
    "High School",
    "Bachelor's",
    "Master's",
    "PhD"
])
job_title = st.text_input("Job Title", "Software Engineer")

if st.button("Predict Salary"):    
    # Prepare the input data dictionary
    new_data = {
        'Age': float(age),
        'Gender': gender,
        'Education Level': education_level,
        'Job Title': job_title,
        'Years of Experience': float(years_of_experience)
    }

    # Display the input data for verification (optional)
    st.subheader("Input Data for Prediction:")
    st.write(pd.DataFrame([new_data]))

    try:
        # Make the prediction
        predicted_val = predict_salary(new_data)
        st.success(f"### Predicted Salary: ${predicted_val:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please ensure all necessary model files (`best_model.pkl`, `scaler.pkl`, `feature_columns.pkl`) are available in the deployment environment and your `predict_salary.py` script is correctly set up.")

st.markdown("---")
st.info("This app uses a pre-trained model to estimate salaries based on the provided inputs.")
