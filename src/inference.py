import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.custom_exception import CustomException
from src.logger import get_logger
from config.path_config import *

logger = get_logger(__name__)


class Inference:
    """
    ML Inference class (MODEL-STAGE ONLY)

    Assumes:
    - Cleaning, filtering, feature extraction already done
    - Feature CSV exists for ONE trial
    """

    def __init__(self, shot_name, feature_root_dir, model_name="knn", player_name=None):
        self.shot_name = shot_name
        self.feature_root_dir = Path(feature_root_dir)
        self.player_name = player_name
        self.model_name = model_name

        self.model_dir = Path(TRAINING_MODELS_DIR) / self.shot_name

    def resolve_feature_csv(self):
        if self.shot_name is None:
            raise RuntimeError("shot_name must be provided for inference")

        shot_dir = self.feature_root_dir / self.shot_name

        if not shot_dir.exists():
            raise FileNotFoundError(f"shot feature directory not found: {shot_dir}")

        csv_files = list(shot_dir.glob("*.csv"))

        if len(csv_files) == 0:
            raise FileNotFoundError(f"No feature CSV found in {shot_dir}")
        if len(csv_files) > 1:
            raise RuntimeError(f"Multiple feature CSVs found in {shot_dir}, expected 1")

        return csv_files[0]


    # ------------------------------------------------------------------
    # LOAD ARTIFACTS (MODEL-ONLY)
    # ------------------------------------------------------------------
    def load_artifacts(self):
        try:
            self.model = joblib.load(
                self.model_dir / f"{self.model_name}_final_model.joblib"
            )
            self.scaler = joblib.load(
                self.model_dir / "scaler.joblib"
            )
            self.label_encoder = joblib.load(
                self.model_dir / "label_encoder.joblib"
            )
            self.feature_cols = pd.read_csv(
                self.model_dir / "feature_columns.csv", header=None
            )[0].tolist()

        except Exception as e:
            raise CustomException(
                f"Failed to load inference artifacts for shot '{self.shot_name}'",
                sys
            )

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------
    def predict(self):
        feature_csv = self.resolve_feature_csv()
        df_feat = pd.read_csv(feature_csv)

        missing = set(self.feature_cols) - set(df_feat.columns)
        if missing:
            raise RuntimeError(f"Missing required features: {sorted(missing)}")

        X = df_feat.loc[:, self.feature_cols]

        if X.shape[1] != self.scaler.n_features_in_:
            raise RuntimeError(
                f"Feature mismatch: scaler expects {self.scaler.n_features_in_}, "
                f"got {X.shape[1]}"
            )

        X_scaled = self.scaler.transform(X)

        pred_idx = self.model.predict(X_scaled)[0]
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]

        probs = None
        if hasattr(self.model, "predict_proba"):
            probs_raw = self.model.predict_proba(X_scaled)[0]
            probs = {
                self.label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(probs_raw)
            }

        return pred_label, probs


    # ------------------------------------------------------------------
    # RUN (PIPELINE ENTRY)
    # ------------------------------------------------------------------
    def run(self):
        self.load_artifacts()
        label, probs = self.predict()

        logger.info(f"Predicted Performance Level: {label}")
        if probs:
            logger.info("Prediction probabilities:")
            for k, v in probs.items():
                logger.info(f"  {k}: {v:.3f}")

        return label, probs


if __name__ == "__main__":
    shot_name = input(
        "Enter shot type "
        "(shortpassshot / curve / longpass / powershot / shortpass / trivela / volley): "
    )
    player_name = input("Enter Playername: ")

    feature_csv = "artifacts/inference/features/shot_name/trial_1_features.csv"

    inference = Inference(shot_name=shot_name,
        player_name=player_name,
        feature_root_dir= INFERENCE_FEATURES_DIR,
        model_name="rf")
    
    inference.run()
