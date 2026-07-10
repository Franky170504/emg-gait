import streamlit as st
import base64
from app.core.cleaning import Cleaning

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


st.markdown("""
<style>

/* Remove top padding */
.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
}

/* Normal buttons */
div.stButton > button{
    width:100%;
    height:68px;
    border-radius:15px;
    font-size:24px;
    font-weight:600;
}

/* Download buttons */
div.stDownloadButton > button{
    width:100%;
    height:68px;              /* Same height as other buttons */
    border-radius:15px;
    font-size:22px;
    font-weight:600;

    border:2px solid #E6E66A;
    background:rgba(0,0,0,0);
    color:#E6E66A;

    transition:0.3s;
}

div.stDownloadButton > button:hover{
    background:rgba(230,230,106,0.15);
    border-color:#FFFF80;
    color:#FFFF80;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Text inside input fields */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    color: #000000 !important;
}

/* Placeholder text */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #9E9E9E !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

[data-testid="stFileUploader"] span[class = "st-emotion-cache-miu686 e3v525e4"] {
    color:#36a7b8;
}

/* Upload button text */
[data-testid="stFileUploader"] button{
    color:#e9ef89 !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* + and - icons */
[data-testid="stNumberInput"] svg{
    fill:#0a0a00 !important;
    color:#0a0a00 !important;
}

/* Hover */
[data-testid="stNumberInput"] button:hover svg{
    fill:#FFFF80 !important;
    color:#FFFF80 !important;
}

</style>
""", unsafe_allow_html=True)

add_bg_from_local("app/media/background_2.png")

st.title("EMG Data Cleaning")

player_name = st.text_input("Enter your name")

shot_number = st.number_input("Enter shot number",step=1, min_value=1)

shot_type = st.radio(
    "Shot Type",
    ["Chipshot", "Curve", "Longpass", "Powershot", "Shortpass", "Trivela", "Volley"],
    horizontal=True
)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    max_upload_size=None

)

if uploaded_file is not None:

    cleaner = Cleaning()
    cleaned_df = cleaner.process_file(uploaded_file)

    st.success("File cleaned successfully!")

    st.subheader("Cleaned Data")
    st.dataframe(cleaned_df)

    muscles = cleaner.get_muscle_list(cleaned_df)

    selected_muscles = st.multiselect(
        "Select muscles to display",
        options=muscles,
        default=muscles  # All selected by default
    )

    if selected_muscles:
        fig = cleaner.create_plot(
            cleaned_df,
            selected_muscles,
        )

        st.plotly_chart(fig, width="stretch")
        st.divider()
        st.subheader("Save Results")

        if player_name and shot_type and shot_number:

            base_filename = f"{player_name}_{shot_type}_{shot_number}"

            # CSV download
            csv = cleaned_df.to_csv(index=False).encode("utf-8")
            svg_bytes = fig.to_image(format="svg")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                label="Download Cleaned CSV",
                data=csv,
                file_name=f"{base_filename}.csv",
                mime="text/csv",
            )
            # Place a button in the second column
            with col2:
                st.download_button(
                label="Download Graph",
                data=svg_bytes,
                file_name=f"{base_filename}.svg",
                mime="image/svg+xml",
            )
        

        
