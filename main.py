import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("best_model.pkl")
encoders = joblib.load("encoders.pkl")  # Dictionary of LabelEncoders

# Filtered/cleaned options (based on your preprocessing)
valid_education = [e for e in encoders['education'].classes_ if e not in ['1st-4th', '5th-6th', 'Preschool']]
valid_workclass = [w for w in encoders['workclass'].classes_ if w not in ['Without-pay', 'Never-worked']]
valid_occupation = list(encoders['occupation'].classes_)
valid_gender = list(encoders['gender'].classes_)

st.set_page_config(page_title="Employee Salary Prediction", layout="wide")
st.title("💼 Employee Salary Classification App")

# Two-column layout
left_col, right_col = st.columns([1, 1])

with left_col:
    st.header("📝 Input Details")

    age = st.slider("Age", min_value=17, max_value=90, value=30)
    education = st.selectbox("Education", options=valid_education)
    workclass = st.selectbox("Workclass", options=valid_workclass)
    hours_per_week = st.slider("Hours per Week", min_value=1, max_value=100, value=40)
    gender = st.radio("Gender", options=valid_gender, horizontal=True)
    occupation = st.selectbox("Occupation", options=valid_occupation)

    if st.button("Predict"):
        # Show human-readable input
        original_input = {
            'Age': age,
            'Education': education,
            'Workclass': workclass,
            'Hours per Week': hours_per_week,
            'Gender': gender,
            'Occupation': occupation
        }

        # Display original input
        with right_col:
            st.header("📊 Prediction Result")

            st.write("### 🔍 Your Input Summary")
            st.table(pd.DataFrame([original_input]))

        # Prepare input for model prediction
        input_data = pd.DataFrame({
            'age': [age],
            'education': [education],
            'workclass': [workclass],
            'hours-per-week': [hours_per_week],
            'gender': [gender],
            'occupation': [occupation]
        })

        # Encode categorical fields
        for col in ['education', 'workclass', 'gender', 'occupation']:
            input_data[col] = encoders[col].transform(input_data[col])

        # Predict
        prediction = model.predict(input_data)[0]
        prediction_label = ">50K" if prediction == 1 else "≤50K"

        with right_col:
            st.success(f"💰 Predicted Salary Class: **{prediction_label}**")

