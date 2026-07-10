import base64
import streamlit as st


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
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}


        /* Hide footer */
        footer {{
            visibility: hidden;
        }}

        /* Remove top padding */
        .block-container {{
            padding-top: 0rem;
            padding-bottom: 0rem;
        }}

        /* Button styling */
        div.stButton > button {{
            width: 100%;
            height: 68px;

            border-radius: 15px;

            border: 2px solid #E6E66A;

            background: rgba(0,0,0,0);

            color: #E6E66A;

            font-size: 24px;
            font-weight: 600;

            transition: 0.3s;
        }}

        div.stButton > button:hover {{
            background: rgba(230,230,106,0.15);
            border-color: #FFFF80;
            color: #FFFF80;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


# Background image
add_bg_from_local("app/media/background.png")


# Vertical spacing so the buttons sit near the bottom
st.markdown("<div style='height:600px'></div>", unsafe_allow_html=True)


# Center the buttons
left, btn1, gap, btn2, right = st.columns([3.0, 1.2, 0.25, 1.2, 1.5])

with btn1:
    if st.button("How it works?", use_container_width=True):
        st.switch_page("pages/working.py")

with btn2:
    if st.button("Start analysis", use_container_width=True):
        st.switch_page("pages/upload.py")