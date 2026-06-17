import streamlit as st
import pandas as pd
import joblib
import subprocess

# Set app title
# st.title("CSV File Drop & Graph Visualizer")

models = {
    "Chipshot": joblib.load(r"artifacts\training\models\chipshot\gradient_boost_final_model.joblib"),
    "Curve": joblib.load(r"artifacts\training\models\curve\knn_final_model.joblib"),
    "Longpass": joblib.load(r"artifacts\training\models\longpass\voting_final_model.joblib"),
    "Powershot": joblib.load(r"artifacts\training\models\powershot\extra_tree_final_model.joblib"),
    "Shortpass": joblib.load(r"artifacts\training\models\shortpass\cat_boost_final_model.joblib"),
    "Trivela": joblib.load(r"artifacts\training\models\trivela\knn_final_model.joblib"),
    "Volley": joblib.load(r"artifacts\training\models\volley\cat_boost_final_model.joblib")
}

player_name = st.text_input("Enter your name")

shot_type = st.radio(
    "Shot Type",
    ["Chipshot", "Curve", "Longpass", "Powershot", "Shortpass", "Trivela", "Volley"],
    horizontal=True
)

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

if st.button("Predict"):

    process = subprocess.run(
        ["python", r"pipe\inference_pipeline.py"],
        input=f"{shot_type}\n{player_name}\n",
        text=True,
        capture_output=True
    )

    if process.returncode == 0:
        st.success("Prediction completed")
        st.text(process.stdout)
    else:
        st.error(process.stderr)
