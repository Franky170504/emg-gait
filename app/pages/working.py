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

st.markdown('<p style="font-size: 20px;">' \
'''Football is a physically demanding sport that requires precise coordination of multiple lower-limb muscles to executedifferent kicking techniques effectively. Each type of kickinvolves distinct biomechanical characteristics and muscleactivation patterns that influence power generation, accuracy,and control. Improper activation of muscles during kicking can lead to inefficient performance and may increase the risk of muscular strain or injury. Despite this, performance evaluation in football training is still largely based on visual observation and subjective judgment by coaches.'''

'<p style="font-size: 20px;">' \
'''Surface electromyography has emerged as a reliable method for directly measuring muscle activation during dynamic movements. EMG signals provide insight into the timing and intensity of muscle activity, making them valuable for biomechanical analysis. When combined with machine learning techniques, EMG data can be used to model complex movement patterns and distinguish between different types of actions.'''

'<p style="font-size: 20px;">' \
'''The key contributions of this work are as follows:-'''

'<p style="font-size: 20px;">' \
'''1)Kick-specific machine learning models are trained to capture the biomechanical characteristics of each kick.'''

'<p style="font-size: 20px;">' \
'''2)A single, repetition evaluation framework is proposed, allowing practical use in real-world scenarios. '''

'<p style="font-size: 20px;">' \
'''3)Muscle-wise deviation analysis is performed to generate interpretable feedback for performance improvement.'''
'</p>', text_alignment="justify", unsafe_allow_html=True)

add_bg_from_local("app/media/background_2.png")
left, btn1, gap, btn2, right = st.columns([3.0, 1.2, 0.25, 1.2, 1.5])
with btn1:
    if st.button("Start analysis", use_container_width=True):
        st.switch_page("pages/upload.py")