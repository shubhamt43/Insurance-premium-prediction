import streamlit as st
import pandas as pd
from predict import predict_output

# City Tier Lists
tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = [
    "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore",
    "Bhopal", "Nagpur", "Vadodara", "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
    "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram", "Ludhiana", "Nashik",
    "Allahabad", "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli",
    "Bhavnagar", "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode", "Warangal",
    "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur", "Asansol", "Siliguri"
]

st.set_page_config(page_title="Insurance Predictor", layout="centered")

st.title("Insurance Premium Category Predictor")
st.markdown("Enter your details below:")

# --- FRONTEND INPUT FIELDS ---
age = st.number_input("Age", min_value=1, max_value=119, value=30)
weight = st.number_input("Weight (kg)", min_value=1.0, value=65.0)
height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7)
income_lpa = st.number_input("Annual Income (LPA)", min_value=0.1, value=10.0)
smoker = st.selectbox("Are you a smoker?", options=[True, False])
city = st.text_input("City", value="Mumbai")
occupation = st.selectbox(
    "Occupation",
    ['retired', 'freelancer', 'student', 'government_job', 'business_owner', 'unemployed', 'private_job']
)

# --- PREDICTION LOGIC ---
if st.button("Predict Premium Category"):
    with st.spinner("Processing data and generating prediction..."):
        try:
            # 1. City Normalization
            normalized_city = city.strip().title()

            # 2. BMI Calculation
            bmi = weight / (height ** 2)

            # 3. Lifestyle Risk Logic (Exact match - LOWERCASE)
            if smoker and bmi > 30:
                lifestyle_risk = "high"
            elif smoker or bmi > 27:
                lifestyle_risk = "medium"
            else:
                lifestyle_risk = "low"

            # 4. Age Group Logic (CRITICAL FIX: Exact match with user_input.py - LOWERCASE)
            if age < 25:
                age_group = "young"
            elif age < 45:
                age_group = "adult"
            elif age < 60:
                age_group = "middle_aged"
            else:
                age_group = "senior"

            # 5. City Tier Logic
            if normalized_city in tier_1_cities:
                city_tier = 1
            elif normalized_city in tier_2_cities:
                city_tier = 2
            else:
                city_tier = 3

            # 6. Final Dictionary matching the model's exact features
            user_input = {
                'bmi': bmi,
                'age_group': age_group,
                'lifestyle_risk': lifestyle_risk,
                'city_tier': city_tier,
                'income_lpa': income_lpa,
                'occupation': occupation
            }

            # 7. Model Prediction
            result = predict_output(user_input)

            # --- DISPLAY RESULTS ---
            st.success(f"Predicted Insurance Premium Category: **{result['predicted_class']}**")
            st.write(f"🔍 **Confidence:** {result['confidence'] * 100:.2f}%")
            
            st.write("📊 **Class Probabilities:**")
            formatted_probs = {k: f"{v * 100:.2f}%" for k, v in result['class_probabilities'].items()}
            st.json(formatted_probs)

        except FileNotFoundError:
            st.error("❌ `model.pkl` file nahi mili! Make sure it is inside a folder named `model` (e.g., `model/model.pkl`).")
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
