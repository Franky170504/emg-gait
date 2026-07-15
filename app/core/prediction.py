import sys
from pathlib import Path
import joblib
import pandas as pd

from src.custom_exception import CustomException
from config.path_config import APP_MODELS_DIR

class Inference:
    def __init__(self, shot_name, player_name=None):
        self.shot_name = shot_name
        self.player_name = player_name
        self.load_artifacts()

    def load_artifacts(self):
        model_root = Path(APP_MODELS_DIR)

        if self.shot_name == "Chipshot":
            folder = model_root / "chipshot"
            model_file = "gradient_boost_final_model.joblib"

        elif self.shot_name == "Curve":
            folder = model_root / "curve"
            model_file = "knn_final_model.joblib"

        elif self.shot_name == "Longpass":
            folder = model_root / "longpass"
            model_file = "voting_final_model.joblib"

        elif self.shot_name == "Powershot":
            folder = model_root / "powershot"
            model_file = "extra_tree_final_model.joblib"

        elif self.shot_name == "Shortpass":
            folder = model_root / "shortpass"
            model_file = "cat_boost_final_model.joblib"

        elif self.shot_name == "Trivela":
            folder = model_root / "trivela"
            model_file = "knn_final_model.joblib"

        elif self.shot_name == "Volley":
            folder = model_root / "volley"
            model_file = "cat_boost_final_model.joblib"

        else:
            raise ValueError(f"Unknown shot type: {self.shot_name}")

        self.model = joblib.load(folder / model_file)
        self.scaler = joblib.load(folder / "scaler.joblib")
        self.label_encoder = joblib.load(folder / "label_encoder.joblib")
        self.feature_columns = pd.read_csv(folder / "feature_columns.csv",header=None)[0].tolist()

    def predict(self, feature_df):
        missing = set(self.feature_columns) - set(feature_df.columns)
        if missing:
            raise RuntimeError(f"Missing features: {sorted(missing)}")

        X = feature_df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]
        label = self.label_encoder.inverse_transform([pred])[0]
        probabilities = None

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)[0]
            probabilities = {
                self.label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(probs)
            }
        return label, probabilities