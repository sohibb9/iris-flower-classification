from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "iris_model.joblib"
SCALER_PATH = BASE_DIR / "iris_scaler.joblib"

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

SPECIES = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica",
}

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container {max-width: 900px; padding-top: 2.2rem; padding-bottom: 3rem;}
      .hero {padding: 1.5rem 1.6rem; border: 1px solid rgba(128,128,128,.25);
             border-radius: 18px; margin-bottom: 1.2rem;
             background: linear-gradient(120deg, rgba(231,170,205,.13), rgba(129,190,174,.10));}
      .hero h1 {margin: 0 0 .35rem 0; font-size: 2.25rem;}
      .hero p {margin: 0; opacity: .78;}
      div.stButton > button {width: 100%; border-radius: 10px; min-height: 2.8rem; font-weight: 650;}
      [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: 12px 14px; border-radius: 12px;}
      .small-note {font-size: .88rem; opacity: .72;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Load trained artifacts
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        missing = [p.name for p in (MODEL_PATH, SCALER_PATH) if not p.exists()]
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))

    trained_model = joblib.load(MODEL_PATH)
    trained_scaler = joblib.load(SCALER_PATH)
    return trained_model, trained_scaler


try:
    model, scaler = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the trained model. Details: {exc}")
    st.info("Make sure iris_model.joblib and iris_scaler.joblib are in the same folder as app.py.")
    st.stop()


# -----------------------------------------------------------------------------
# App content
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🌸 Iris Flower Classifier</h1>
      <p>Enter four flower measurements and let the machine learning model predict its species.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Flower measurements")
st.caption("All measurements are in centimeters (cm).")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input(
            "Sepal length (cm)", min_value=0.0, max_value=10.0,
            value=5.1, step=0.1, format="%.1f"
        )
        petal_length = st.number_input(
            "Petal length (cm)", min_value=0.0, max_value=10.0,
            value=1.4, step=0.1, format="%.1f"
        )

    with col2:
        sepal_width = st.number_input(
            "Sepal width (cm)", min_value=0.0, max_value=10.0,
            value=3.5, step=0.1, format="%.1f"
        )
        petal_width = st.number_input(
            "Petal width (cm)", min_value=0.0, max_value=10.0,
            value=0.2, step=0.1, format="%.1f"
        )

    submitted = st.form_submit_button("Predict flower species 🌼")

if submitted:
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=FEATURES,
    )

    try:
        scaled_data = scaler.transform(input_data)
        predicted_class = int(model.predict(scaled_data)[0])
        probabilities = model.predict_proba(scaled_data)[0]

        predicted_name = SPECIES.get(predicted_class, f"Class {predicted_class}")
        confidence = float(np.max(probabilities)) * 100

        st.divider()
        st.subheader("Prediction result")
        result_col, confidence_col = st.columns(2)
        with result_col:
            st.success(f"Predicted species: **{predicted_name}**")
        with confidence_col:
            st.metric("Model confidence", f"{confidence:.2f}%")

        st.subheader("Probability by species")
        probability_df = pd.DataFrame(
            {
                "Species": [SPECIES.get(int(class_id), str(class_id)) for class_id in model.classes_],
                "Probability": probabilities,
            }
        ).sort_values("Probability", ascending=False)

        chart_df = probability_df.set_index("Species")
        st.bar_chart(chart_df, x_label="Species", y_label="Probability")
        st.dataframe(
            probability_df.assign(Probability=probability_df["Probability"].map(lambda x: f"{x:.2%}")),
            hide_index=True,
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.markdown("**About this project**")
st.write(
    "This app uses a Logistic Regression classifier trained on the classic Iris dataset. "
    "The four measurements are standardized using the saved StandardScaler before prediction."
)
st.markdown(
    '<p class="small-note">Educational machine learning demo · Built with Python, scikit-learn and Streamlit</p>',
    unsafe_allow_html=True,
)
