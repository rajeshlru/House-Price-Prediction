import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="HouseVision AI", layout="wide")

st.write("App Loaded Successfully 🚀")

# ---------------- SAFE LOAD FILES ----------------
try:
    model = joblib.load("house_model.pkl")
    scaler = joblib.load("scaler.pkl")
    df = pd.read_csv("dataset.csv")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("🏠 HouseVision AI")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Predict", "Analytics", "History"]
)

# HOME

if page == "Home":

    st.title("🏠 HouseVision AI")
    st.subheader("AI-powered property valuation platform")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset Size", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1])
    with col3:
        st.metric("Average Price", f"₹ {int(df['Price'].mean()):,}")

    st.markdown("---")
    st.info("👉 Navigate to 'Predict' to estimate house prices instantly.")


# PREDICT

elif page == "Predict":

    st.title("🔮 Predict House Price")

    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("Number of Bedrooms", 1, 10, 3)
        bathrooms = st.number_input("Number of Bathrooms", 1, 10, 2)
        living_area = st.number_input("Living Area (sq ft)", 200, 10000, 1500)
        floors = st.number_input("Number of Floors", 1, 5, 2)

    with col2:
        grade = st.number_input("Grade of House", 1, 13, 7)
        basement = st.number_input("Basement Area", 0, 5000, 500)
        condition = st.number_input("Condition of House", 1, 5, 3)

    if st.button("Predict Price 🚀"):

        input_data = np.array([[ 
            bedrooms,
            bathrooms,
            living_area,
            floors,
            grade,
            basement,
            condition
        ]])

        input_scaled = scaler.transform(input_data)

        prediction_log = model.predict(input_scaled)[0]
        prediction = np.expm1(prediction_log)

        st.session_state.history.append(prediction)

        st.success(f"Estimated Price: ₹ {prediction:,.0f}")

# ANALYTICS

elif page == "Analytics":

    st.title("📊 Housing Analytics")

    fig = px.histogram(df, x="Price", nbins=30, title="Price Distribution")
    st.plotly_chart(fig, use_container_width=True)

# HISTORY

elif page == "History":

    st.title("🕒 Prediction History")

    if st.session_state.history:
        history_df = pd.DataFrame(
            st.session_state.history,
            columns=["Predicted Price"]
        )
        st.dataframe(history_df)
    else:
        st.info("No predictions made yet.")