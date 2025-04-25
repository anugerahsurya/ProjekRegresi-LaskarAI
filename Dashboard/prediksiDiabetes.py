import os
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load encoder
with open("Dashboard/gender_label_encoder.pkl", 'rb') as file:
    gender_encoder = pickle.load(file)

# Load scaler
with open("Dashboard/minmax_scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

# Load model
with open("Dashboard/BestModelCB.pkl", 'rb') as file:
    model = pickle.load(file)


# Kolom one-hot smoking history
smoking_ohe_columns = [
    "smoking_history_No Info",
    "smoking_history_current",
    "smoking_history_ever",
    "smoking_history_former",
    "smoking_history_never",
    "smoking_history_not current"
]

# Fungsi hitung BMI
def hitung_bmi(berat, tinggi_cm):
    tinggi_m = tinggi_cm / 100
    return berat / (tinggi_m ** 2)

# Judul
st.title("Prediksi Diabetes")

# Form input
with st.form("health_form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", min_value=0, max_value=120, value=25)
    hypertension = st.number_input("Riwayat Hipertensi (0 = Tidak, 1 = Ya)", min_value=0, max_value=1)
    heart_disease = st.number_input("Riwayat Penyakit Hati (0 = Tidak, 1 = Ya)", min_value=0, max_value=1)
    
    # Input berat & tinggi
    berat_badan = st.number_input("Berat Badan (kg)", min_value=1.0, step=0.1)
    tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=30.0, step=0.1)
    
    hba1c_level = st.number_input("HbA1c Level", min_value=0.0, step=0.1)
    blood_glucose_level = st.number_input("Level Gula Darah", min_value=0.0, step=0.1)
    smoking_history = st.selectbox(
        "Riwayat Merokok", 
        ["No Info", "current", "ever", "former", "never", "not current"]
    )

    submitted = st.form_submit_button("Submit")

    if submitted:
        # Hitung BMI
        bmi = hitung_bmi(berat_badan, tinggi_badan)

        # Encode gender
        gender_encoded = gender_encoder.transform([gender])[0]

        # One-hot encoding untuk smoking_history
        smoking_ohe = [1 if f"smoking_history_{smoking_history}" == col else 0 for col in smoking_ohe_columns]

        # Gabung semua fitur
        features = [
            gender_encoded,
            age,
            hypertension,
            heart_disease,
            bmi,
            hba1c_level,
            blood_glucose_level
        ] + smoking_ohe

        features_array = np.array([features])

        # Normalisasi
        scaled_features = scaler.transform(features_array)

        # Prediksi
        prediction = model.predict(scaled_features)[0]
        pred_proba = model.predict_proba(scaled_features)[0][1]

        # Output
        st.success("Data berhasil diproses!")
        st.write("### Hasil Prediksi Diabetes:")
        st.write(f"**Prediksi:** {'Rentan Diabetes' if prediction == 1 else 'Tidak Diabetes'}")
        if prediction == 1:
            st.write("Kamu memiliki risiko terkena diabetes. Untuk itu Kamu harus dapat menjaga kadar gula darah di tubuh dengan mengurangi konsumsi makanan dan minuman yang mengandung gula berlebih. Selain itu biasakan untuk mengonsumsi makanan yang sehat.")
        else:
            st.write("Jaga selalu kesehatan Kamu yaa.")
