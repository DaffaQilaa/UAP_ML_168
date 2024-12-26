import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Load the pre-trained model
@st.cache_resource
def load_text_model():
    return tf.keras.models.load_model(r"src/hotel_sentiment.h5")

# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    with open(r"src/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load stopwords
@st.cache_resource
def load_stopwords():
    with open(r"src/stopwords.pkl", "rb") as f:
        stopwords = pickle.load(f)
    return stopwords

# Text preprocessing function
def preprocess_text(text, tokenizer, stopwords, max_len=100):
    # Lowercase and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stopwords])

    # Tokenize and pad sequences
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded

# Streamlit UI
st.title("Analisis Sentimen Ulasan Hotel")
st.write("Masukkan ulasan hotel untuk mengetahui prediksi rating keseluruhan berdasarkan sentimen.")

# Load resources
model = load_text_model()
tokenizer = load_tokenizer()
stopwords = load_stopwords()

input_text = st.text_area("Masukkan Ulasan Hotel", "")
if st.button("Prediksi Rating"):
    if input_text:
        preprocessed_text = preprocess_text(input_text, tokenizer, stopwords)
        prediction = model.predict(preprocessed_text)
        st.write(f"Prediction Output: {prediction}")  # Debugging output model
        rating = round(prediction[0][0] * 5, 2)  # Convert to a 5-star scale
        st.write(f"Prediksi Rating: {rating} dari 5")
    else:
        st.warning("Harap masukkan ulasan.")