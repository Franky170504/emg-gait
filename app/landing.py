# app/landing.py

from pathlib import Path
import sys
import streamlit as st
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="EMG Gait", layout="wide")

st.title("EMG Gait Analysis")
st.write("Redirecting...")

time.sleep(1)

st.switch_page("pages/uploadpage.py")