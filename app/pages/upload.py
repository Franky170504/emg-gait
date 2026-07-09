import io

import streamlit as st
from app.core.cleaning import DataCleaning

st.markdown("""
<style>

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
    color: #1565C0 !important;
}

/* Placeholder text */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #9E9E9E !important;
}

</style>
""", unsafe_allow_html=True)

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
    type=["csv"]
)

if uploaded_file is not None:

    cleaner = DataCleaning()
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
            selected_muscles
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
        

        
