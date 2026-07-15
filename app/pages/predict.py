import streamlit as st
import base64

from app.core.filtering import Filtering
from app.core.feature_extraction import FeatureExtraction
from app.core.prediction import Inference

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

st.set_page_config(
    page_title="Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("Prediction Results")

# ------------------------------------------------------------------
# Check session state
# ------------------------------------------------------------------

required_keys = [
    "cleaned_df",
    "player_name",
    "shot_type",
    "shot_number"
]

for key in required_keys:
    if key not in st.session_state:
        st.error("No uploaded data found. Please upload a file first.")
        st.stop()

# ------------------------------------------------------------------
# Read stored values
# ------------------------------------------------------------------

cleaned_df = st.session_state["cleaned_df"]
player_name = st.session_state["player_name"]
shot_name = st.session_state["shot_type"]
shot_number = st.session_state["shot_number"]

# ------------------------------------------------------------------
# Run pipeline
# ------------------------------------------------------------------

try:
    with st.spinner("Running prediction..."):

        # Filtering
        filtering = Filtering()
        filtered_df = filtering.process_dataframe(
            cleaned_df
        )
        # Feature Extraction
        feature_extractor = FeatureExtraction()
        feature_df = feature_extractor.process_dataframe(
            filtered_df
        )
        # Inference
        inference = Inference(
            shot_name=shot_name,
            player_name=player_name
        )
        prediction, probabilities = inference.predict(
            feature_df
        )

except Exception as e:
    st.error(f"Prediction failed.\n\n{e}")
    st.stop()

# ------------------------------------------------------------------
# Display Results
# ------------------------------------------------------------------

st.success("Prediction Completed Successfully!")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Player", player_name)
with col2:
    st.metric("Shot Type", shot_name)
with col3:
    st.metric("Shot Number", shot_number)

st.divider()
st.subheader("Predicted Performance")
st.metric(
    label="Performance Level",
    value=prediction
)

# ------------------------------------------------------------------
# Prediction Probabilities
# ------------------------------------------------------------------

if probabilities is not None:
    st.divider()
    st.subheader("Prediction Confidence")
    st.bar_chart(probabilities)

# ------------------------------------------------------------------
# Optional Details
# ------------------------------------------------------------------

with st.expander("Filtered Signal"):
    st.dataframe(
        filtered_df,
        width="stretch"
    )

with st.expander("Extracted Features"):
    st.dataframe(
        feature_df,
        width="stretch"
    )

# ------------------------------------------------------------------
# Back Button
# ------------------------------------------------------------------

if st.button("⬅ Back to Upload"):
    st.switch_page("pages/upload.py")