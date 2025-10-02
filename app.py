# app.py -- Streamlit UI for Wellness Package Predictor (single + batch)
# app.py - Simple Wellness Tourism Package predictor (single input form)
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.title("Wellness Tourism Package Predictor")
st.write("Predict whether a customer will purchase the Wellness Tourism Package.")

# Paths to artifacts (edit if different)
MODEL_PATH = Path("/content/drive/MyDrive/TourismPackage/models/best_model.joblib")
PREPROC_PATH = Path("/content/drive/MyDrive/TourismPackage/preprocessor.joblib")

# --- Input fields (main area) ---
st.subheader("Enter customer details")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3], index=1)
duration_of_pitch = st.number_input("Duration of Pitch (mins)", min_value=0, max_value=300, value=10)
occupation = st.text_input("Occupation", "Salaried")
gender = st.selectbox("Gender", ["Male", "Female"])
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=0, max_value=20, value=2)
num_followups = st.number_input("Number of Followups", min_value=0, max_value=20, value=1)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Wellness"], index=3)
preferred_star = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=4)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=1)
num_trips = st.number_input("Number of Trips (per year)", min_value=0, max_value=50, value=2)
passport = st.selectbox("Has Passport?", [0, 1], index=1)
own_car = st.selectbox("Own Car?", [0, 1], index=0)
num_children = st.number_input("Number of Children Visiting (<5 yrs)", min_value=0, max_value=10, value=0)
designation = st.text_input("Designation", "Executive")
monthly_income = st.number_input("Monthly Income", min_value=0, max_value=10_000_000, value=50000)

# Predict button
if st.button("Predict Purchase"):
    # Build input dataframe (columns must match those used when preprocessor was fitted)
    input_dict = {
        'Age': age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_person_visiting,
        "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": num_trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": num_children,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    st.success(f"Purchase : {prediction}")
    


