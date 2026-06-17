import streamlit as st

name = st.text_input("Enter your name")


genre = st.radio(
    "Shot Type",
    ["Chipshot", "Curve", "Longpass", "Powershot", "Shortpass", "Trivela", "Volley"],
    horizontal=True
)

st.button("Predict")