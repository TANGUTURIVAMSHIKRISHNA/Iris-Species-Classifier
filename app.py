import streamlit as st
import pandas as pd
import pickle

# Load model, scaler, and label encoder
model = pickle.load(open('iris_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# Streamlit app
st.title("Iris Species Classifier")

# Input sliders for features
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Predict
if st.button("Predict"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    species = le.inverse_transform(prediction)[0]
    st.write(f"Predicted Species: **{species}**")


