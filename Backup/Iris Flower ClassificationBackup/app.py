import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page settings
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")


# Load saved model and scaler
@st.cache_resource
def load_model():
    folder = Path(__file__).resolve().parent
    model = joblib.load(folder / "iris_model.joblib")
    scaler = joblib.load(folder / "iris_scaler.joblib")
    return model, scaler


model, scaler = load_model()

# Page title
st.title("🌸 Iris Flower Classifier")

st.write("Enter flower measurements to predict its species.")

# User inputs
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2)

# Predict button
if st.button("Predict Flower"):
    features = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    flower = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]], columns=features
    )

    flower_scaled = scaler.transform(flower)

    prediction = model.predict(flower_scaled)[0]
    probabilities = model.predict_proba(flower_scaled)[0]

    names = ["Setosa", "Versicolor", "Virginica"]

    st.success(f"Predicted Species: {names[prediction]}")

    st.subheader("Prediction Probabilities")

    probability_df = pd.DataFrame(
        {"Species": [names[i] for i in model.classes_], "Probability": probabilities}
    )

    st.bar_chart(probability_df.set_index("Species"))
