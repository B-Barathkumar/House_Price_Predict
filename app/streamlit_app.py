import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# --------------------------------------------------------------
# FIX: XGBoost compatibility patch for old trained models
# --------------------------------------------------------------
try:
    # Removes deprecated 'use_label_encoder' and aligns defaults
    xgb.XGBClassifier.__init__.__defaults__ = (
        None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None,
        None, None, None, False  # <- last parameter was use_label_encoder
    )
except:
    pass


# --------------------------------------------------------------
# LOAD MODELS + DATASET
# --------------------------------------------------------------
preprocessor = joblib.load("models/preprocessor.pkl")
investment_model = joblib.load("models/investment_classifier.pkl")
price_model = joblib.load("models/price_regressor.pkl")

# Load dataset for Market Insights
df = pd.read_excel("data/india_housing_prices.xlsx")


# --------------------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide",
)

st.title("🏠 Real Estate Investment Advisor")
st.write("Analyze property investment potential and forecast future prices using Machine Learning.")


# --------------------------------------------------------------
# USER INPUT FORM
# --------------------------------------------------------------
st.sidebar.header("📋 Enter Property Details")

def user_input_features():

    state = st.sidebar.selectbox("State", sorted(df["State"].unique()))
    city = st.sidebar.selectbox("City", sorted(df[df["State"] == state]["City"].unique()))
    locality = st.sidebar.selectbox("Locality", sorted(df[df["City"] == city]["Locality"].unique()))

    property_type = st.sidebar.selectbox("Property Type", df["Property_Type"].unique())
    bhk = st.sidebar.slider("BHK", 1, 6, 3)

    size_sqft = st.sidebar.number_input("Size (SqFt)", min_value=300, max_value=5000, value=1200)
    price_lakhs = st.sidebar.number_input("Price (Lakhs)", min_value=5, max_value=500, value=75)

    year_built = st.sidebar.number_input("Year Built", min_value=1970, max_value=2025, value=2020)
    furnished_status = st.sidebar.selectbox("Furnished Status", df["Furnished_Status"].unique())

    floor_no = st.sidebar.slider("Floor No", 0, 50, 2)
    total_floors = st.sidebar.slider("Total Floors", 1, 60, 10)

    nearby_schools = st.sidebar.slider("Nearby Schools (count)", 0, 20, 3)
    nearby_hospitals = st.sidebar.slider("Nearby Hospitals (count)", 0, 20, 2)

    public_transport = st.sidebar.selectbox("Public Transport Accessibility", df["Public_Transport_Accessibility"].unique())
    parking_space = st.sidebar.selectbox("Parking Space", df["Parking_Space"].unique())
    security = st.sidebar.selectbox("Security", df["Security"].unique())
    amenities = st.sidebar.selectbox("Amenities", df["Amenities"].unique())
    facing = st.sidebar.selectbox("Facing", df["Facing"].unique())
    owner_type = st.sidebar.selectbox("Owner Type", df["Owner_Type"].unique())
    availability = st.sidebar.selectbox("Availability Status", df["Availability_Status"].unique())

    data = {
        "State": state,
        "City": city,
        "Locality": locality,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size_sqft,
        "Price_in_Lakhs": price_lakhs,
        "Year_Built": year_built,
        "Furnished_Status": furnished_status,
        "Floor_No": floor_no,
        "Total_Floors": total_floors,
        "Nearby_Schools": nearby_schools,
        "Nearby_Hospitals": nearby_hospitals,
        "Public_Transport_Accessibility": public_transport,
        "Parking_Space": parking_space,
        "Security": security,
        "Amenities": amenities,
        "Facing": facing,
        "Owner_Type": owner_type,
        "Availability_Status": availability,
    }

    return pd.DataFrame([data])

user_data = user_input_features()


# --------------------------------------------------------------
# FEATURE ENGINEERING (same as training)
# --------------------------------------------------------------
user_data["Price_per_SqFt"] = (user_data["Price_in_Lakhs"] * 100000) / user_data["Size_in_SqFt"]
user_data["Age_of_Property"] = 2025 - user_data["Year_Built"]

user_data["School_Density_Score"] = user_data["Nearby_Schools"] / user_data["Size_in_SqFt"]
user_data["Hospital_Density_Score"] = user_data["Nearby_Hospitals"] / user_data["Size_in_SqFt"]

city_pps = df[df["City"] == user_data["City"][0]]["Price_per_SqFt"].median()
user_data["City_Median_PPS"] = city_pps


# --------------------------------------------------------------
# PREDICTIONS
# --------------------------------------------------------------
st.subheader("🔍 Prediction Results")

try:
    processed = preprocessor.transform(user_data)

    invest_pred = investment_model.predict(processed)[0]
    invest_prob = investment_model.predict_proba(processed)[0][1]
    future_price = price_model.predict(processed)[0]

    st.success(f"🏡 Investment Recommendation: {'GOOD INVESTMENT' if invest_pred == 1 else 'NOT RECOMMENDED'}")
    st.write(f"📊 Model Confidence: {invest_prob * 100:.2f}%")
    st.info(f"💰 Predicted Future Price (5 Years): ₹ {future_price:.2f} Lakhs")

except Exception as e:
    st.error("⚠️ Error occurred during prediction.")
    st.exception(e)


# --------------------------------------------------------------
# MARKET INSIGHTS
# --------------------------------------------------------------
st.subheader("📊 Market Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("### Median Price by Top Cities")
    city_prices = (
        df.groupby("City")["Price_in_Lakhs"]
        .median()
        .sort_values(ascending=False)
        .head(15)
    )
    st.bar_chart(city_prices)

with col2:
    st.write("### Price per SqFt Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["Price_per_SqFt"], kde=True, bins=50, color="green", ax=ax)
    ax.set_xlabel("Price per SqFt (₹)")
    ax.set_ylabel("Number of Properties")
    ax.ticklabel_format(style='plain', axis='x')
    st.pyplot(fig)
