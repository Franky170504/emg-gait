import os
import json
import joblib
import warnings
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Machine Learning Imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# Project Internal Imports
from src.custom_exception import CustomException
from src.logger import get_logger
from config.path_config import *

# Setup
logger = get_logger(__name__)
warnings.filterwarnings("ignore")

class ModelTraining:
    def __init__(self, clusters_dir):
        """
        :param clusters_dir: Path to the clusters folder (e.g., data/clusters)
        """
        self.clusters_root = Path(clusters_dir)
        # Create 'models' directory at the same level as the clusters directory
        self.models_root = self.clusters_root.parent / "models"
        self.models_root.mkdir(parents=True, exist_ok=True)
        
        self.RANDOM_STATE = 42
        self.BASE_RF_N_ESTIMATORS = 500
        self.TARGET_FILENAME = "features_master_labeled_gmm4.csv"

        logger.info(f"Initialized ModelTraining. Clusters: {self.clusters_root} | Models: {self.models_root}")

    def get_models(self, n_classes):
        """Initializes the algorithms to be trained for each shot."""
        models = {
            'rf': RandomForestClassifier(n_estimators=self.BASE_RF_N_ESTIMATORS, 
                                         class_weight='balanced', random_state=self.RANDOM_STATE, n_jobs=-1),
            # 'lr': LogisticRegression(multi_class='multinomial', max_iter=2000, 
            #                          class_weight='balanced', random_state=self.RANDOM_STATE),
            'svm': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=self.RANDOM_STATE),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'xgb': xgb.XGBClassifier(objective='multi:softprob', num_class=n_classes, n_estimators=200, 
                                     eval_metric='mlogloss', random_state=self.RANDOM_STATE)
        }
        return models

    def prepare_data(self, csv_path):
        """Loads CSV and extracts features, labels, and player IDs."""
        df = pd.read_csv(csv_path)
        label_col = 'performance_level_adjusted' if 'performance_level_adjusted' in df.columns else 'performance_level'
        
        exclude_meta = {'file', 'player', 'n_samples', 'fs_used', 'gmm_label', 'gmm_confidence',
                        'performance_level', 'performance_level_confidence', 'performance_level_adjusted', '_proxy_score'}
        
        # 1. Select only numeric columns (safely handles pandas specific types)
        valid_feature_cols = df.select_dtypes(include=[np.number]).columns

        # 2. Filter out any metadata columns that happened to be numeric
        feature_cols = sorted([c for c in valid_feature_cols if c not in exclude_meta])        
        X = df[feature_cols].fillna(df[feature_cols].median()).values.astype(float)
        y = df[label_col].astype(str).values
        players = df['player'].astype(str).values
        
        le = LabelEncoder().fit(y)
        return X, le.transform(y), players, le, feature_cols

    def process_shot_folder(self, shot_cluster_path):
        """Processes a specific shot: training and saving into a mirrored models subdirectory."""
        shot_name = shot_cluster_path.name
        
        # Mirror the shot folder structure inside the 'models' directory
        shot_models_output_dir = self.models_root / shot_name
        shot_models_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f">>> STARTING: {shot_name}")
        
        X, y_enc, players, le, feature_cols = self.prepare_data(shot_cluster_path / self.TARGET_FILENAME)
        unique_players = np.unique(players)
        
        if len(unique_players) < 2:
            logger.warning(f"Skipping {shot_name}: Insufficient players for LOPO validation.")
            return

        models_dict = self.get_models(len(le.classes_))
        lopo_results = {name: {"y_true": [], "y_pred": []} for name in models_dict.keys()}

        # 1. LOPO Validation Loop
        for fold_idx, test_player in enumerate(unique_players, start=1):
            test_mask = (players == test_player)
            train_mask = ~test_mask
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y_enc[train_mask], y_enc[test_mask]

            scaler = StandardScaler().fit(X_train)
            X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

            for name, clf in models_dict.items():
                clf_fold = clone(clf).fit(X_train_s, y_train)
                lopo_results[name]['y_true'].extend(y_test)
                lopo_results[name]['y_pred'].extend(clf_fold.predict(X_test_s))

        # 2. Final Training and Metrics Export
        scaler_final = StandardScaler().fit(X)
        X_final_s = scaler_final.transform(X)
        
        for name, clf in models_dict.items():
            # Save final model for each algorithm
            final_clf = clone(clf).fit(X_final_s, y_enc)
            joblib.dump(final_clf, shot_models_output_dir / f"{name}_final_model.joblib")
            
            # Generate and save individual performance reports
            y_t, y_p = np.array(lopo_results[name]['y_true']), np.array(lopo_results[name]['y_pred'])
            report = classification_report(y_t, y_p, target_names=le.classes_, zero_division=0)
            
            perf_content = (f"Algorithm: {name}\nShot Category: {shot_name}\n"
                            f"Overall Accuracy: {accuracy_score(y_t, y_p):.4f}\n"
                            f"F1-Macro Score: {f1_score(y_t, y_p, average='macro', zero_division=0):.4f}\n\n"
                            f"Detailed Report:\n{report}")
            
            with open(shot_models_output_dir / f"{name}_metrics.txt", "w") as f:
                f.write(perf_content)
            
            logger.info(f"Shot: {shot_name} | Model: {name} | F1: {f1_score(y_t, y_p, average='macro'):.4f}")

        # 3. Save Shared Pipeline Artifacts
        joblib.dump(scaler_final, shot_models_output_dir / "scaler.joblib")
        joblib.dump(le, shot_models_output_dir / "label_encoder.joblib")
        pd.Series(feature_cols).to_csv(shot_models_output_dir / "feature_columns.csv", index=False)
        
        logger.info(f"<<< COMPLETED: {shot_name}. Results saved to {shot_models_output_dir}")

    def run(self):
        """Iterates through all cluster subdirectories and executes training."""
        # Find directories inside clusters_dir that have the target CSV
        shot_folders = [p.parent for p in self.clusters_root.rglob(self.TARGET_FILENAME)]
        logger.info(f"Detected {len(shot_folders)} shot categories to process.")

        for shot_path in shot_folders:
            try:
                self.process_shot_folder(shot_path)
            except Exception as e:
                logger.error(f"Error in {shot_path.name}: {str(e)}")
                raise CustomException(e, sys)

if __name__ == "__main__":
    # CLUSTERS_DIR is passed from path_config.py
    trainer = ModelTraining(TRAINING_CLUSTERS_DIR)
    trainer.run()