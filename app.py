import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("models/xgb_delay_predictor.pkl")

# App settings
st.set_page_config(
    page_title="Flight Delay Predictor", 
    page_icon="✈️",
    layout="wide"
)

# Header
st.title("✈️ Flight Delay Minutes Predictor")
st.markdown("""
### Predict total monthly delay minutes for any Airline–Airport combination  
This tool uses a trained **XGBoost ML model** on U.S. aviation delay data.
""")

st.markdown("---")

# Layout: Left inputs, Right info
left_col, right_col = st.columns([2,1])

with left_col:
    st.subheader("🛫 Input Details")
    
    carrier = st.selectbox(
        "Airline Carrier Code",
        ['AA','AS','B6','DL','EV','F9','G4','HA','MQ','NK','OO','UA','WN','YV','9E','QX','US'],
        help="Select the airline whose delay you want to predict."
    )

    airport = st.selectbox(
        "Airport Code",
        sorted([
            'ATL','LAX','ORD','DFW','DEN','JFK','SFO','SEA','LAS','MCO','CLT','PHX',
            'MIA','EWR','IAH','BOS','MSP','DTW','FLL','PHL','BWI','SLC','SAN','MDW',
            'TPA','DCA','PDX','HNL','AUS'
        ]),
        help="Choose the airport for your prediction."
    )

    season = st.selectbox(
        "Season",
        ["winter", "spring", "summer", "fall"],
        help="Season influences air traffic & weather patterns."
    )

    year = st.number_input(
        "Year", min_value=2010, max_value=2024, value=2018,
        help="Enter the year for which you want to estimate delays."
    )

    month = st.number_input(
        "Month", min_value=1, max_value=12, value=1,
        help="Select the month (1–12)."
    )

    arr_flights = st.number_input(
        "Arrival Flights",
        min_value=1, max_value=10000, value=500,
        help="Total incoming flights for that airline and airport."
    )

    arr_cancelled = st.number_input(
        "Cancelled Flights",
        min_value=0, max_value=1000, value=10,
        help="Number of cancelled flights that month."
    )

    arr_diverted = st.number_input(
        "Diverted Flights",
        min_value=0, max_value=1000, value=5,
        help="Number of flights diverted to another airport."
    )

    delay_rate = st.slider(
        "Historical Delay Rate",
        min_value=0.0, max_value=1.0, value=0.20,
        help="Percentage of delayed flights historically."
    )

    total_delay_causes = st.number_input(
        "Total Delay Causes",
        min_value=0, max_value=3000, value=150,
        help="Total number of contributing delay events recorded."
    )

with right_col:
    st.subheader("📘 About the Model")
    st.info("""
    **Model:** XGBoost Regressor  
    **Trained On:** 170,000+ real airline delay records  
    **Goal:** Predict total delay minutes in a given month  
    """)
    st.success("""
    ✔ Handles categorical + numerical features  
    ✔ Accurate R² score: **94.7%**  
    ✔ Suitable for aviation analytics  
    """)

st.markdown("---")

# Predict Button
if st.button("🔮 Predict Delay Minutes"):
    
    input_data = pd.DataFrame({
        "year": [year],
        "month": [month],
        "arr_flights": [arr_flights],
        "arr_cancelled": [arr_cancelled],
        "arr_diverted": [arr_diverted],
        "delay_rate": [delay_rate],
        "total_delay_causes": [total_delay_causes],
        "carrier": [carrier],
        "airport": [airport],
        "season": [season]
    })
    
    prediction = model.predict(input_data)[0]

    st.subheader("🧾 Prediction Result")
    st.success(f"Estimated Total Delay Minutes: **{prediction:.2f} minutes**")

    # Convert to hours & days
    hours = prediction / 60
    days = hours / 24

    st.info(f"""
    **Breakdown:**  
    - **Hours:** {hours:.2f}  
    - **Days:** {days:.2f}
    """)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + XGBoost | Aviation ML Project")
