import streamlit as st
from streamlit_option_menu import option_menu

# Fungsi untuk menjalankan berbagai halaman

def klasifikasi_tabular():
    st.page("tabular.py")
    st.title("Klasifikasi Data Tabular")
    st.write("Ini adalah halaman untuk klasifikasi data tabular.")

def klasifikasi_text():
    st.page("teks.py")
    st.title("Klasifikasi Data Tek")
    st.write("Untuk analisis dataset teks")
