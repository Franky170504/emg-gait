import streamlit as st
import base64

def set_background_video(video_file):
    # Read the video file and encode it to base64
    with open(video_file, "rb") as f:
        video_bytes = f.read()
    
    video_base64 = base64.b64encode(video_bytes).decode('utf-8')
    
    # CSS to make the video a full-screen background
    css = f"""
    <style>
    /* 1. Ensure the video is at the very back */
    .background-video {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        object-fit: cover;
        z-index: -1; /* Pushes video behind everything */
    }}

    /* 2. Make the main Streamlit container transparent */
    [data-testid="stAppViewContainer"] {{
        background: transparent !important;
    }}
    
    /* 3. Ensure the main block is positioned above the video */
    [data-testid="stMainBlockContainer"] {{
        position: relative;
        z-index: 1;
        background: rgba(255, 255, 255, 0.2); /* Semi-transparent background for readability */
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    
    <video class="background-video" autoplay loop muted playsinline>
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
    """
    st.markdown(css, unsafe_allow_html=True)

st.markdown("""
    <style>
    .custom-text {
        font-size: 28px !important;
        font-family: 'Arial', sans-serif !important;
        font-weight: bold !important;
        text-align: center !important;
        color: #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

# Call the function
set_background_video(r"artifacts\media\bg.mp4")

# Add your app content here
# 2. Add your content
st.title("Welcome to My App")
st.markdown("This text appears on top of the video.")

if st.button("Upload Files"):
    st.switch_page(r"pages/visual.py")
