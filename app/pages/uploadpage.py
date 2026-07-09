import streamlit as st
from app.core.cleaning import DataCleaning

st.title("EMG Data Cleaning")

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
    else:
        st.warning("Please select at least one muscle.")
