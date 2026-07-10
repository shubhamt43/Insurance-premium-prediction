import streamlit as st
import pandas as pd
import joblib  # .pkl file load karne ke liye

# --- CONFIGURATION & MODEL LOADING ---
st.set_page_config(page_title="Insurance Predictor", layout="centered")

@st.cache_resource
def load_my_model():
    # joblib se aapki trained model.pkl file directly load ho jayegi
    try:
        model = joblib.load("model.pkl")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model.pkl: {str(e)}")
        return None

# Model ko memory me load karein (st.cache_resource se ye baar-baar load nahi hoga)
my_model = load_my_model()

# --- FRONTEND UI ---
st.title("Insurance Premium Category Predictor")
st.markdown("Enter your details below to predict the premium category:")

# Input fields (Aapke frontend ke mutabik)
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
    if my_model is None:
        st.error("Model is not loaded. Please check your model.pkl file.")
    else:
        with st.spinner("Processing data and predicting..."):
            try:
                # 1. Feature Engineering / Preprocessing (Same as your model's expected inputs)
                bmi = weight / (height ** 2)
                
                if age < 25:
                    age_group = "Youth"
                elif age < 60:
                    age_group = "Adult"
                else:
                    age_group = "Senior"
                    
                city_tier = "Tier 1" if city.lower() in ['mumbai', 'delhi', 'bangalore', 'kolkata'] else "Tier 2"
                lifestyle_risk = "High" if smoker else "Low"
                
                # 2. Creating DataFrame matching your model's exact feature names and structure
                # Note: Make sure columns order matches exactly how your model was trained!
                input_df = pd.DataFrame([{
                    'bmi': bmi,
                    'age_group': age_group,
                    'lifestyle_risk': lifestyle_risk,
                    'city_tier': city_tier,
                    'income_lpa': income_lpa,
                    'occupation': occupation
                }])
                
                # 3. Model Prediction
                # Agar aapka model direct classification output deta hai:
                prediction = my_model.predict(input_df)[0]
                
                # Display Result
                st.success(f"Predicted Insurance Premium Category: **{prediction}**")
                
                # Optional: Agar aapka model probabilities support karta hai (predict_proba)
                if hasattr(my_model, "predict_proba"):
                    probabilities = my_model.predict_proba(input_df)[0]
                    st.write("📊 Class Probabilities:")
                    # Display probabilities in a nice dictionary layout
                    classes = my_model.classes_
                    prob_dict = {str(c): f"{p*100:.2f}%" for c, p in zip(classes, probabilities)}
                    st.json(prob_dict)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Tip: Double-check if the input features and categorical values match your training data.")
