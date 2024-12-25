import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model_path = r'src/ular.h5'
model = load_model(model_path)

# Expected features from the trained model
expected_features = [
    'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
    'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'
]

# Streamlit app title
st.title("Aplikasi Klasifikasi Data Tabular - Banking Dataset")

# User input
st.write("Masukkan data untuk diklasifikasikan:")

# Collect user inputs
age = st.number_input("Usia", min_value=18, max_value=100, value=30)
job = st.selectbox("Pekerjaan", options=['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                                          'retired', 'student', 'technician', 'services', 'unemployed', 'unknown'])
marital = st.selectbox("Status Perkawinan", options=['divorced', 'married', 'single', 'unknown'])
education = st.selectbox("Pendidikan", options=['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox("Apakah memiliki kredit macet?", options=['yes', 'no'])
balance = st.number_input("Saldo", value=0.0)
housing = st.selectbox("Apakah memiliki pinjaman perumahan?", options=['yes', 'no'])
loan = st.selectbox("Apakah memiliki pinjaman?", options=['yes', 'no'])
contact = st.selectbox("Tipe Kontak", options=['cellular', 'telephone'])
day = st.number_input("Hari Terakhir Dihubungi", min_value=1, max_value=31, value=1)
month = st.selectbox("Bulan Terakhir Dihubungi", options=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
                                                           'aug', 'sep', 'oct', 'nov', 'dec'])
duration = st.number_input("Durasi Kontak (detik)", value=0)
campaign = st.number_input("Jumlah Kontak dalam Kampanye", value=1)
pdays = st.number_input("Jumlah Hari Sejak Kontak Terakhir", value=-1)  # -1 jika tidak pernah dihubungi
previous = st.number_input("Jumlah Kontak Sebelumnya", value=0)
poutcome = st.selectbox("Hasil Kontak Sebelumnya", options=['failure', 'nonexistent', 'success'])

# When the classify button is clicked
if st.button("Klasifikasikan"):
    # Create DataFrame from inputs
    input_data = pd.DataFrame([[age, job, marital, education, default, balance, housing, loan, contact,
                                 day, month, duration, campaign, pdays, previous, poutcome]],
                               columns=['age', 'job', 'marital', 'education', 'default', 'balance',
                                        'housing', 'loan', 'contact', 'day', 'month', 'duration',
                                        'campaign', 'pdays', 'previous', 'poutcome'])

    # One-hot encoding for categorical features
    input_data = pd.get_dummies(input_data, columns=['job', 'marital', 'education', 'default',
                                                     'housing', 'loan', 'contact', 'month', 'poutcome'], drop_first=True)

    # Add missing columns with zero and ensure correct column order
    for col in expected_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[expected_features]

    # Convert to numpy array
    input_array = input_data.to_numpy(dtype=np.float32)

    # Validate input dimensions
    if input_array.shape[1] != len(expected_features):
        st.error(f"Jumlah fitur input tidak sesuai: {input_array.shape[1]} ditemukan, {len(expected_features)} diharapkan.")
    else:
        # Predict with the model
        prediction = model.predict(input_array)
        result = "Ya" if prediction[0] > 0.5 else "Tidak"
        st.write(f"Hasil Klasifikasi: Apakah pelanggan akan berlangganan deposito berjangka? {result}")
