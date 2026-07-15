import streamlit as st
import base64

st.set_page_config(
    page_title="EMG Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>

        /* Background */
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/svg;base64,{encoded_string}");
            background-size: cover;
            background-position: fill;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

         </style>
        """,
        unsafe_allow_html=True,
    )

add_bg_from_local("app/media/background_2.png")

left, btn1, gap, btn2, right = st.columns([3.0, 1.2, 0.25, 1.2, 1.5])

with btn1:
    if st.button("Save Results"):
        st.switch_page("pages/upload.py")

