import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# --- 1. Path Setup for 'src' location ---
# Get the Project Root directory (parent of 'src')
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add Project Root to sys.path so we can import 'config' and 'src' modules
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- 2. Imports ---
try:
    # Now we can import from config because PROJECT_ROOT is in path
    from config.path_config import MODELS_DIR
    # Import siblings from src
    from src.custom_exception import CustomException
    from src.logger import get_logger
    logger = get_logger(__name__)
except ImportError as e:
    # Fallback for standalone testing
    import logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
    logger = logging.getLogger("Inference")
    logger.warning(f"Import error ({e}). Using fallback configuration.")
    
    # Fallback paths if config is missing
    MODELS_DIR = PROJECT_ROOT / "data" / "models"
    
    class CustomException(Exception):
        def __init__(self, error_message, error_detail):
            super().__init__(error_message)

warnings.filterwarnings("ignore")

class ModelInference:
    def __init__(self, models_root_path):
        """
        :param models_root_path: Path object to the root 'models' directory.
        """
        self.models_root = Path(models_root_path)
        
        if not self.models_root.exists():
            raise FileNotFoundError(f"Models directory not found at: {self.models_root}")

    def load_artifacts(self, shot_name, algorithm='rf'):
        """
        Loads the trained model, scaler, label encoder, and feature list.
        """
        # Construct path: data/models/chipshot
        shot_model_dir = self.models_root / shot_name
        
        required_files = {
            "model": shot_model_dir / f"{algorithm}_final_model.joblib",
            "scaler": shot_model_dir / "scaler.joblib",
            "le": shot_model_dir / "label_encoder.joblib",
            "features": shot_model_dir / "feature_columns.csv"
        }
        
        for name, path in required_files.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing artifact for '{shot_name}' at: {path}")
        
        try:
            artifacts = {
                "model": joblib.load(required_files["model"]),
                "scaler": joblib.load(required_files["scaler"]),
                "le": joblib.load(required_files["le"]),
                "feature_cols": pd.read_csv(required_files["features"]).iloc[:, 0].tolist()
            }
            return artifacts
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_csv_path, shot_name=None, algorithm='rf', output_path=None):
        try:
            logger.info(f"Reading input data from {input_csv_path}...")
            df = pd.read_csv(input_csv_path)
            
            # Auto-detect shot type if not provided
            if shot_name is None:
                if 'shot_type' in df.columns:
                    shot_name = df['shot_type'].mode()[0]
                    logger.info(f"Auto-detected shot type: {shot_name}")
                else:
                    raise ValueError("Could not detect 'shot_type'. Please provide it explicitly.")
            
            # Load artifacts
            artifacts = self.load_artifacts(shot_name, algorithm)
            model = artifacts["model"]
            scaler = artifacts["scaler"]
            le = artifacts["le"]
            feature_cols = artifacts["feature_cols"]
            
            # Validate Columns
            missing_cols = [c for c in feature_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(f"Input CSV missing columns: {missing_cols[:3]}...")
            
            # Prepare Features (Exact order match)
            X = df[feature_cols].copy()
            X = X.fillna(X.median()).fillna(0) # Handle NaNs
            
            # Scale & Predict
            X_scaled = scaler.transform(X.values)
            preds = le.inverse_transform(model.predict(X_scaled))
            
            # Build Result
            results = df.copy()
            results['predicted_performance'] = preds
            
            if hasattr(model, "predict_proba"):
                results['prediction_confidence'] = np.max(model.predict_proba(X_scaled), axis=1)
                
            if output_path:
                results.to_csv(output_path, index=False)
                logger.info(f"Saved results to {output_path}")
            
            return results

        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    # --- CONFIGURATION BASED ON YOUR FILE STRUCTURE ---
    
    # 1. Models Directory: Defined in config, or deduced relative to project root
    # Should point to: Project_Root/data/models
    CURRENT_MODELS_DIR = Path(MODELS_DIR) if 'MODELS_DIR' in locals() else PROJECT_ROOT / "data" / "models"
    
    # 2. Shot Name: As per your diagram, the data is inside 'chipshot'
    TARGET_SHOT = "chipshot"

    # 3. Input File: Located inside data/models/chipshot/ as per your diagram
    INPUT_FILE = CURRENT_MODELS_DIR / TARGET_SHOT / "features_master_labeled_gmm4.csv"
    
    # 4. Output File: Saving it to the same directory as input for convenience
    OUTPUT_FILE = INPUT_FILE.parent / "inference_output.csv"
    
    ALGO = 'rf'

    print(f"--- Starting Inference ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Models Path:  {CURRENT_MODELS_DIR}")
    print(f"Input File:   {INPUT_FILE}")
    
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at {INPUT_FILE}")
    else:
        try:
            inferencer = ModelInference(CURRENT_MODELS_DIR)
            
            results = inferencer.predict(
                input_csv_path=INPUT_FILE, 
                shot_name=TARGET_SHOT, # Explicitly passing it since we know the folder
                algorithm=ALGO,
                output_path=OUTPUT_FILE
            )
            
            print("\nSuccess! Results Preview:")
            print(results[['player', 'predicted_performance', 'prediction_confidence']].head())
            
        except Exception as e:
            print(f"\nExecution Failed: {e}")