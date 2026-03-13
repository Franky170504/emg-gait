import sys
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

from src.custom_exception import CustomException
from src.logger import get_logger
from config.path_config import *

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


class ModelTraining:
    """
    FINAL, CONSISTENT ML TRAINING PIPELINE
    """

    TARGET_FILENAME = "features_master_labeled_gmm4.csv"
    RANDOM_STATE = 42

    def __init__(self, clusters_dir):
        self.clusters_root = Path(clusters_dir)
        self.models_root = self.clusters_root.parent / "models"
        self.models_root.mkdir(parents=True, exist_ok=True)

        logger.info(f"Clusters dir: {self.clusters_root}")
        logger.info(f"Models dir: {self.models_root}")

    # ------------------------------------------------------------------
    # MODELS
    # ------------------------------------------------------------------
    def get_models(self, n_classes):

    # ---------- Base Models ----------
        rf_model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=self.RANDOM_STATE,
            n_jobs=-1
        )

        svm_model = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=self.RANDOM_STATE
        )

        knn_model = KNeighborsClassifier(
            n_neighbors=5
        )

        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=500,
            eval_metric="mlogloss",
            random_state=self.RANDOM_STATE
        )

        extra_tree_model = ExtraTreesClassifier(
            n_estimators=500,
            random_state=self.RANDOM_STATE,
            n_jobs=-1
        )

        gradient_boost_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05
        )

        cat_boost_model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            verbose=0
        )

        gnb_model = GaussianNB()

        # ---------- Ensemble Models ----------

        voting_model = VotingClassifier(
            estimators=[
                ("rf", rf_model),
                ("svm", svm_model),
                ("xgb", xgb_model)
            ],
            voting="soft",
            n_jobs=-1
        )

        stacking_model = StackingClassifier(
            estimators=[
                ("rf", rf_model),
                ("svm", svm_model),
                ("xgb", xgb_model),
                ("knn", knn_model)
            ],
            final_estimator=LogisticRegression(max_iter=2000),
            n_jobs=-1
        )

        # ---------- Model Dictionary ----------
        return {
            "rf": rf_model,
            "svm": svm_model,
            "knn": knn_model,
            "xgb": xgb_model,
            "extra_tree": extra_tree_model,
            "gradient_boost": gradient_boost_model,
            "cat_boost": cat_boost_model,
            # "lda": lda_model,
            # "qda": qda_model,
            "gaussian_nb": gnb_model,
            "voting": voting_model,
            "stacking": stacking_model
        }

    # ------------------------------------------------------------------
    # DATA PREPARATION
    # ------------------------------------------------------------------
    def prepare_data(self, csv_path):
        df = pd.read_csv(csv_path)

        label_col = (
            "performance_level_adjusted"
            if "performance_level_adjusted" in df.columns
            else "performance_level"
        )

        exclude_meta = {
            "file", "player", "n_samples", "fs_used",
            "gmm_label", "gmm_confidence",
            "performance_level",
            "performance_level_confidence",
            "performance_level_adjusted",
            "_proxy_score"
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = sorted([c for c in numeric_cols if c not in exclude_meta])

        X = df[feature_cols].fillna(df[feature_cols].median())
        y = df[label_col].astype(str)
        players = df["player"].astype(str)

        le = LabelEncoder().fit(y)

        return (
            X.values,
            le.transform(y),
            players.values,
            le,
            feature_cols
        )

    # ------------------------------------------------------------------
    # TRAIN PER SHOT
    # ------------------------------------------------------------------
    def train_shot(self, shot_cluster_dir):
        shot_name = shot_cluster_dir.name
        logger.info(f"\n===== TRAINING SHOT: {shot_name} =====")

        csv_path = shot_cluster_dir / self.TARGET_FILENAME
        model_dir = self.models_root / shot_name
        model_dir.mkdir(parents=True, exist_ok=True)

        X, y, players, le, feature_cols = self.prepare_data(csv_path)
        unique_players = np.unique(players)

        if len(unique_players) < 2:
            logger.warning(f"Skipping {shot_name}: not enough players for LOPO.")
            return

        models = self.get_models(len(le.classes_))
        lopo_results = {k: {"y_true": [], "y_pred": []} for k in models}

        # ------------------------------
        # LOPO VALIDATION
        # ------------------------------
        for test_player in unique_players:
            test_mask = players == test_player
            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            scaler = StandardScaler().fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)

            for name, clf in models.items():
                clf_fold = clone(clf).fit(X_train_s, y_train)
                lopo_results[name]["y_true"].extend(y_test)
                lopo_results[name]["y_pred"].extend(clf_fold.predict(X_test_s))

        # ------------------------------
        # FINAL TRAINING (AUTHORITATIVE)
        # ------------------------------
        scaler_final = StandardScaler().fit(X)
        X_final_s = scaler_final.transform(X)

        for name, clf in models.items():
            final_model = clone(clf).fit(X_final_s, y)

            joblib.dump(final_model, model_dir / f"{name}_final_model.joblib")

            y_true = np.array(lopo_results[name]["y_true"])
            y_pred = np.array(lopo_results[name]["y_pred"])

            report = classification_report(
                y_true, y_pred,
                target_names=le.classes_,
                zero_division=0
            )

            metrics_text = (
                f"Shot: {shot_name}\n"
                f"Model: {name}\n"
                f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n"
                f"Macro-F1: {f1_score(y_true, y_pred, average='macro'):.4f}\n\n"
                f"{report}"
            )

            with open(model_dir / f"{name}_metrics.txt", "w") as f:
                f.write(metrics_text)

            logger.info(
                f"{shot_name} | {name} | "
                f"F1={f1_score(y_true, y_pred, average='macro'):.4f}"
            )

        # ------------------------------
        # SAVE SHARED ARTIFACTS
        # ------------------------------
        joblib.dump(scaler_final, model_dir / "scaler.joblib")
        joblib.dump(le, model_dir / "label_encoder.joblib")
        pd.Series(feature_cols).to_csv(
            model_dir / "feature_columns.csv", index=False
        )

        logger.info(f"Completed shot: {shot_name}")

    # ------------------------------------------------------------------
    # RUN ALL SHOTS
    # ------------------------------------------------------------------
    def run(self):
        shot_dirs = [p.parent for p in self.clusters_root.rglob(self.TARGET_FILENAME)]
        logger.info(f"Detected {len(shot_dirs)} shot categories.")

        for shot_dir in shot_dirs:
            try:
                self.train_shot(shot_dir)
            except Exception as e:
                logger.error(f"Error in {shot_dir.name}: {str(e)}")
                raise CustomException(e, sys)


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    trainer = ModelTraining(TRAINING_CLUSTERS_DIR)
    trainer.run()
