import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model_path = r'src/ular.h5'
model = load_model(model_path)

# Expected features from the trained model
expected_features = [
    'service', 'cleanliness', 'value', 'location', 'sleep_quality',
    'rooms', 'check_in_front_desk', 'bussiness_service', 'date_stayed', 'via_mobile'
]

# Streamlit app title
st.title("Aplikasi Penilaian Hotel - Prediksi Rating Keseluruhan")

# User input
st.write("Masukkan data penilaian masing-masing aspek berikut:")

# Collect user inputs
service = st.slider("Layanan (Service)", min_value=1, max_value=5, value=3)
cleanliness = st.slider("Kebersihan (Cleanliness)", min_value=1, max_value=5, value=3)
value = st.slider("Nilai (Value)", min_value=1, max_value=5, value=3)
location = st.slider("Lokasi (Location)", min_value=1, max_value=5, value=3)
sleep_quality = st.slider("Kualitas Tidur (Sleep Quality)", min_value=1, max_value=5, value=3)
rooms = st.slider("Kamar (Rooms)", min_value=1, max_value=5, value=3)
check_in_front_desk = st.slider("Check-in & Front Desk", min_value=1, max_value=5, value=3)
bussiness_service = st.slider("Layanan Bisnis (Business Service)", min_value=1, max_value=5, value=3)
date_stayed = st.number_input("Tanggal Menginap (Tanggal format angka)", min_value=1, max_value=31, value=15)
via_mobile = st.selectbox("Apakah rating melalui aplikasi mobile?", options=['yes', 'no'])

# Convert 'via_mobile' to binary
def convert_via_mobile(value):
    return 1 if value == 'yes' else 0

via_mobile = convert_via_mobile(via_mobile)

# When the predict button is clicked
if st.button("Prediksi Rating Keseluruhan"):
    # Create DataFrame from inputs
    input_data = pd.DataFrame([[
        service, cleanliness, value, location, sleep_quality, rooms,
        check_in_front_desk, bussiness_service, date_stayed, via_mobile
    ]], columns=expected_features)

    # Ensure input matches expected features
    input_array = input_data.to_numpy(dtype=np.float32)

    # Validate input dimensions
    if input_array.shape[1] != len(expected_features):
        st.error(f"Jumlah fitur input tidak sesuai: {input_array.shape[1]} ditemukan, {len(expected_features)} diharapkan.")
    else:
        # Predict with the model
        prediction = model.predict(input_array)
        overall_rating = round(prediction[0][0], 2)
        st.write(f"Perkiraan Rating Keseluruhan: {overall_rating} dari 5")