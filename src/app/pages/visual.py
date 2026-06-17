import streamlit as st
import pandas as pd

# Set app title
st.title("CSV File Drop & Graph Visualizer")

# 1. Create a drag-and-drop file uploader area
uploaded_files = st.file_uploader("Drop your CSV file here",accept_multiple_files=True, type=["csv"])

for i, uploaded_file in enumerate(uploaded_files):
    df = pd.read_csv(uploaded_file)
    
    # 3. Show raw data preview (optional)
    st.subheader("Data Preview")
    st.dataframe(df)
    
    # Extract numerical columns for plotting    
    numeric_columns = df.drop(columns="Time").columns.tolist()
    
    if len(numeric_columns) > 0:
        st.subheader("Graph Configurations")
        
        # 4. User selection for axes
        x_axis = df["Time"]
        y_axis = st.selectbox(
            f"Select Y-axis for {uploaded_file.name}",
            numeric_columns,
            key=f"y_axis_{i}"
        )
        
        # Set the dataframe index to X-axis so Streamlit plots it correctly
        plot_data = df.set_index(x_axis)[[y_axis]]
        
        # 5. Render the chosen graph
        st.subheader("Generated Graph")
        st.line_chart(plot_data)
       
    else:
        st.error("The uploaded CSV does not contain any numerical columns to plot.")

