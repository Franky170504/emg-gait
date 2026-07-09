import streamlit as st
from app.core.cleaning import DataCleaning

st.title("EMG Data Cleaning")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    cleaner = DataCleaning()

    try:
        cleaned_df = cleaner.process_file(uploaded_file)

        st.success("File cleaned successfully!")

        st.subheader("Cleaned Data")
        st.dataframe(cleaned_df)
        numeric_columns = cleaned_df.drop(columns="Time").columns.tolist()
    
        if len(numeric_columns) > 0:
            st.subheader("Graph Configurations")
            
            # 4. User selection for axes
            x_axis = cleaned_df["Time"]
            y_axis = st.selectbox(
                f"Select Y-axis for {uploaded_file.name}",
                numeric_columns,
                key=f"y_axis_{i}"
            )
            
            # Set the dataframe index to X-axis so Streamlit plots it correctly
            plot_data = cleaned_df.set_index(x_axis)[[y_axis]]
            
            # 5. Render the chosen graph
            st.subheader("Generated Graph")
            st.line_chart(plot_data)
        
        else:
            st.error("The uploaded CSV does not contain any numerical columns to plot.")

    except Exception as e:
        st.error(e)