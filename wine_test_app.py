import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

# Load the trained pipeline
with open('rf_pca_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

st.title("🍷 Wine Quality Predictor")
st.write("Enter the wine's characteristics to predict its quality:")

# Input: 11 basic features
fixed_acidity = st.number_input("Fixed Acidity", min_value=0.0, step=0.1)
volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, step=0.01)
citric_acid = st.number_input("Citric Acid", min_value=0.0, step=0.01)
residual_sugar = st.number_input("Rsesidual Sugar", min_value=0.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, step=0.0001, format="%.4f")
free_so2 = st.number_input("Free Sulfur Dioxide", min_value=0.0, step=1.0)
total_so2 = st.number_input("Total Sulfur Dioxide", min_value=0.0, step=1.0)
density = st.number_input("Density", min_value=0.0, step=0.0001, format="%.4f")
pH = st.number_input("pH", min_value=0.0, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=0.0, step=0.1)

# Calculate engineered features
alcohol_pH = alcohol * pH
sulfate_chloride_ratio = sulphates / (chlorides + 0.0001)
total_acidity = fixed_acidity + volatile_acidity
free_so2_ratio = free_so2 / (total_so2 + 0.0001)
id = 1

# Final feature vector (in the exact order the model expects, excluding 'Id')
input_data = np.array([[
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_so2,
    total_so2,
    density,
    pH,
    sulphates,
    alcohol,
    id,
    alcohol_pH,
    sulfate_chloride_ratio,
    total_acidity,
    free_so2_ratio
]])

if st.button("Predict Wine Quality"):
    prediction = pipeline.predict(input_data)
    probabilities = pipeline.predict_proba(input_data)
    confidence = np.max(probabilities) * 100
    predicted_class = prediction[0]

    st.success(f"🍷 Predicted Wine Quality: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")

    # Plotly bar chart for confidence scores
    classes = pipeline.classes_
    probs = probabilities[0]

    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=probs,
            marker_color='lightseagreen',
            text=[f"{p:.2f}" for p in probs],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title="Prediction Confidence Distribution",
        xaxis_title="Wine Quality",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        template="plotly_white"
    )

    st.plotly_chart(fig)
