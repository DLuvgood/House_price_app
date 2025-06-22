import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("house_model.pkl")

model = load_model()

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.markdown("Enter details below to get an estimated price (in lakhs).")

area = st.slider("Area (sq ft)", 500, 3000, 1200)
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4])
bathrooms = st.selectbox("Bathrooms", [1, 2, 3])
parking = st.radio("Parking Spots", [0, 1, 2])
location_rating = st.slider("Location Rating (1–10)", 1, 10, 7)

if st.button("Predict Price"):
    features = np.array([[area, bedrooms, bathrooms, parking, location_rating]])
    predicted_price = model.predict(features)[0]
    st.success(f"💰 Estimated Price: ₹{predicted_price:.2f} lakhs")