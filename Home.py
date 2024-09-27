

import streamlit as st

# Inject custom CSS for background color
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color:#d3cbb7;  /* Light grey background */
}

[data-testid="stSidebar"] {
    background-color: #fdf6e3;  /* Sidebar background */
}

[data-testid="stHeader"] {
    background-color: transparent;
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title of the web app
st.markdown("<h1 style='font-size:12vh; text-align: center; padding-bottom: 0; margin-bottom: 0;color: white;'>Monsieur Prédicteur</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='font-size:2.5vh; text-align: center; font-style: italic;color: white;'>your trusted forecaster</h2>", unsafe_allow_html=True)

# Divider
st.divider()

# Instructions
st.write("Navigate through the sidebar to perform different tasks:")
st.write("1. Upload/Connect to Database")
st.write("2. Data Cleaning and Feature Selection")
st.write("3. Forecasting and Download Options")




























