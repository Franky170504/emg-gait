import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os

def generate_distinct_dummy_files(source_file, output_folder, time_col_name):
    """
    Generates 5 DISTINCT dummy files.
    Ensures variance between files by randomly selecting from the top 3 neighbors
    and applying random interpolation.
    """
    print(f"Reading source file: {source_file}...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(source_file)
    except FileNotFoundError:
        print(f"Error: The file '{source_file}' was not found.")
        return

    # 2. Validation
    if time_col_name not in df.columns:
        print(f"Error: Column '{time_col_name}' not found.")
        return

    # Separate Time and Feature Data
    time_data = df[time_col_name].values
    feature_df = df.drop(columns=[time_col_name])
    
    try:
        feature_data = feature_df.astype(float).values
    except ValueError:
        print("Error: Non-time columns must be numeric.")
        return

    # 3. Fit Nearest Neighbors
    # We increase K to 3. This finds the 3 closest points (plus the point itself).
    # This gives the script 'options' to choose from, creating variety between files.
    pool_of_neighbors = 3 
    print(f"Fitting model (Finding top {pool_of_neighbors} closest neighbors for every row)...")
    
    nn_model = NearestNeighbors(n_neighbors=pool_of_neighbors + 1, algorithm='auto')
    nn_model.fit(feature_data)
    _, indices = nn_model.kneighbors(feature_data)

    # 4. Create Output Folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 5. Generation Loop
    print("Generating 5 distinct output files...")
    
    for file_idx in range(1, 6):
        new_features = np.zeros_like(feature_data)
        rows, cols = feature_data.shape

        for i in range(rows):
            # VARIANCE STEP 1: Randomly pick one of the top 3 neighbors
            # indices[i][0] is the point itself.
            # indices[i][1] is closest, [i][2] is 2nd closest, etc.
            # We pick a random index between 1 and pool_of_neighbors
            random_k = np.random.randint(1, pool_of_neighbors + 1)
            neighbor_idx = indices[i][random_k]
            
            original_point = feature_data[i]
            neighbor_point = feature_data[neighbor_idx]
            
            # VARIANCE STEP 2: Random low-sensitivity lambda
            # Moves the point 1% to 15% towards the selected neighbor
            lam = np.random.uniform(0.01, 0.15)
            
            # Calculate new point
            new_point = original_point + lam * (neighbor_point - original_point)
            new_features[i] = new_point

        # Reassemble
        dummy_df = pd.DataFrame(new_features, columns=feature_df.columns)
        dummy_df.insert(0, time_col_name, time_data)

        # Save
        output_filename = f"{FOLDER_DIR}_{file_idx}.csv"
        full_output_path = os.path.join(output_folder, output_filename)
        dummy_df.to_csv(full_output_path, index=False)
        
        # Validation Print: Show the first data point of the first column to prove variance
        first_val = dummy_df.iloc[0, 1] 
        print(f" -> Saved File {file_idx}. (Sample Value Row 0: {first_val:.4f})")

    print("\nProcess Complete! Files saved in:", output_folder)

# --- CONFIGURATION SECTION ---
# 1. Update this to your actual file name
SOURCE_FILE = r'Notebooks\data\processed\ahesan_1.csv' 
FOLDER_DIR = r'dummy31'
# 2. This is the new folder where files will be saved
OUTPUT_DIR = r"Notebooks\data\processed"
# 3. Update this to match your Time column header exactly
TIME_COLUMN = 'Time'            

if __name__ == "__main__":
    generate_distinct_dummy_files(SOURCE_FILE, OUTPUT_DIR, TIME_COLUMN)