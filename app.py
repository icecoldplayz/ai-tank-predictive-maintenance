import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tank AI Assistant", layout="centered")

st.title("AI Tank Maintenance Assistant")
st.write("This AI system helps predict corrosion levels and recommends when a tank should be cleaned.")

# --- SESSION STATE ---
if "step" not in st.session_state:
    st.session_state.step = 0

if "data" not in st.session_state:
    st.session_state.data = {}

# --- MODEL ---
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000

    data = pd.DataFrame({
        "age": np.random.randint(1, 20, n),
        "material": np.random.choice([0, 1, 2], n),
        "temperature": np.random.uniform(10, 50, n),
        "humidity": np.random.uniform(20, 100, n),
        "chemical_exposure": np.random.uniform(0, 10, n),
        "usage_frequency": np.random.randint(1, 50, n),
        "last_cleaned_days": np.random.randint(1, 365, n)
    })

    data["corrosion_level"] = (
        data["age"] * 3 +
        data["humidity"] * 0.4 +
        data["chemical_exposure"] * 4 +
        data["last_cleaned_days"] * 0.08 +
        data["temperature"] * 0.5 +
        data["usage_frequency"] * 0.2
)

    data["corrosion_level"] += np.random.normal(0, 5, n)
    data["corrosion_level"] = np.clip(data["corrosion_level"], 0, 100)

    X = data.drop("corrosion_level", axis=1)
    y = data["corrosion_level"]

    model = RandomForestRegressor()
    model.fit(X, y)
    return model

model = train_model()

# --- PROGRESS BAR ---
total_steps = 7
progress = st.session_state.step / total_steps
st.progress(progress)

# --- STEPS UI ---
if st.session_state.step == 0:
    st.subheader("Step 1: Tank Age")
    st.info("Older tanks are more likely to experience corrosion due to long-term wear and environmental exposure.")

    age = st.number_input("How old is the tank (years)?", 1, 20)

    if st.button("Next"):
        st.session_state.data["age"] = age
        st.session_state.step += 1

elif st.session_state.step == 1:
    st.subheader("Step 2: Material")
    st.info("Different materials corrode at different rates. Steel typically corrodes faster than plastic.")

    material = st.selectbox("What is the tank made of?", ["Steel", "Aluminum", "Plastic"])

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back"):
            st.session_state.step -= 1

    with col2:
        if st.button("Next"):
            material_map = {"Steel": 0, "Aluminum": 1, "Plastic": 2}
            st.session_state.data["material"] = material_map[material]
            st.session_state.step += 1

elif st.session_state.step == 2:
    st.subheader("Step 3: Temperature")
    st.info("Higher temperatures can accelerate chemical reactions, increasing corrosion risk.")

    temp = st.number_input("Average temperature (°C)", 10, 50)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back"):
            st.session_state.step -= 1

    with col2:
        if st.button("Next"):
            st.session_state.data["temperature"] = temp
            st.session_state.step += 1

elif st.session_state.step == 3:
    st.subheader("Step 4: Humidity")
    st.info("High humidity increases moisture, which is a major contributor to corrosion.")

    humidity = st.number_input("Humidity (%)", 20, 100)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back"):
            st.session_state.step -= 1
    with col2:
        if st.button("Next"):
            st.session_state.data["humidity"] = humidity
            st.session_state.step += 1

elif st.session_state.step == 4:
    st.subheader("Step 5: Chemical Exposure")
    st.info("Exposure to chemicals significantly speeds up corrosion depending on their reactivity.")

    chem = st.number_input("Chemical exposure level (0–10)", 0.0, 10.0)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back"):
            st.session_state.step -= 1
    with col2:
        if st.button("Next"):
            st.session_state.data["chemical_exposure"] = chem
            st.session_state.step += 1

elif st.session_state.step == 5:
    st.subheader("Step 6: Usage Frequency")
    st.info("Frequent usage may increase wear and stress on the tank, affecting its condition.")

    usage = st.number_input("Uses per week", 1, 50)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"):
            st.session_state.step -= 1
    with col2:
        if st.button("Next"):
            st.session_state.data["usage_frequency"] = usage
            st.session_state.step += 1

elif st.session_state.step == 6:
    st.subheader("Step 7: Last Cleaning")
    st.info("The longer a tank goes without cleaning, the higher the buildup and corrosion risk.")

    days = st.number_input("Days since last cleaning", 1, 365)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"):
            st.session_state.step -= 1
    with col2:
        if st.button("Analyze Tank"):
            st.session_state.data["last_cleaned_days"] = days
            st.session_state.step += 1

elif st.session_state.step == 7:
    st.subheader("Analysis Result")

    input_df = pd.DataFrame([st.session_state.data])
    prediction = model.predict(input_df)[0]

    st.metric("Predicted Corrosion Level", f"{prediction:.2f}%")

    if prediction > 60:
        st.error("High corrosion detected. Cleaning is strongly recommended.")
        st.write("This tank is at risk of structural degradation and should be inspected soon.")
    else:
        st.success("Corrosion level is within a safe range.")
        st.write("No immediate maintenance is required, but continue monitoring over time.")

    if st.button("Start New Analysis"):
        st.session_state.step = 0
        st.session_state.data = {}