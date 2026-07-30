import base64
import streamlit as st
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
 
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

            /* Button styling */
            div.stDownloadButton > button {{
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

            div.stDownloadButton > button:hover {{
                background: rgba(230,230,106,0.15);
                border-color: #FFFF80;
                color: #FFFF80;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Background image
add_bg_from_local("media/background_1.png")
# Vertical spacing so the buttons sit near the bottom
st.markdown("<div style='height:600px'></div>", unsafe_allow_html=True)

@st.cache_data
def create_zip(folder: Path) -> bytes:
    zip_buffer = BytesIO()

    with ZipFile(zip_buffer, "w", ZIP_DEFLATED) as zip_file:
        for file_path in folder.rglob("*"):
            if file_path.is_file():
                # Preserve the folder structure inside the ZIP
                archive_path = file_path.relative_to(folder.parent)
                zip_file.write(file_path, archive_path)

    return zip_buffer.getvalue()

folder_path = Path(__file__).parent / "csv"


left, btn1, gap, btn2, gap, btn3, right = st.columns([3.0, 1.2, 0.25, 1.2, 0.25, 1.2, 1.5])

with btn1:
    if st.button("How it works?", use_container_width=True):
        st.switch_page("pages/working.py")

with btn2:
    if st.button("Start analysis", use_container_width=True):
        st.switch_page("pages/upload.py")

with btn3:
    if folder_path.exists():
        st.download_button(
            label="Download sample files",
            data=create_zip(folder_path),
            file_name="sample_folder.zip",
            mime="application/zip",
        )
    else:
        st.error("Sample folder was not found.")

