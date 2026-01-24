import os
os.environ["OMP_NUM_THREADS"] = "1"
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import zscore

from src.custom_exception import CustomException
from src.logger import get_logger
from config.path_config import *

logger = get_logger(__name__)

class Standardize_Cluster:
    def __init__(self, feature_dir, clusters_dir):
        self.feature_dir = Path(feature_dir)
        self.cluster_dir = Path(clusters_dir)
        self.current_shot_name = None 
        
        self.N_COMPONENTS_PCA = 3
        self.GMM_COMPONENTS = 4
        self.RANDOM_STATE = 42
        self.AUTO_FEATURE_KEYWORDS = [
            "__rms", "__peak", "__iEMG", "__mnf", "__mdf", "__time_to_peak", "__dur_halfmax",
            "__mrms_mean", "__mrms_peak", "coact", "LR_mean_ratio", "__std", "__wl"
        ]

        # Dynamically find all shot subfolders that contain a features_master.csv
        self.shot_paths = [p.parent for p in self.feature_dir.rglob("features_master.csv")]
        logger.info(f"Found {len(self.shot_paths)} shot folders: {[p.name for p in self.shot_paths]}")

    def fit_GMM(self, shot_folder):
        """Standardizes features and fits GMM for a specific shot type."""
        self.current_shot_name = shot_folder.name
        feature_csv_path = shot_folder / "features_master.csv"
        
        # FIX: Read from the local folder path passed in, not self.feature_csv
        self.df = pd.read_csv(feature_csv_path)
        logger.info(f"Processing {self.current_shot_name}: {self.df.shape}")

        # Select candidate features
        candidate_cols = []
        for col in self.df.columns:
            for kw in self.AUTO_FEATURE_KEYWORDS:
                if kw.lower() in col.lower():
                    candidate_cols.append(col)
                    break
        
        if not candidate_cols:
            exclude = {'file','player','n_samples','fs_used'}
            candidate_cols = [c for c in self.df.columns if c not in exclude and np.issubdtype(self.df[c].dtype, np.number)]
        
        self.candidate_cols = sorted(list(set(candidate_cols)))

        # Prepare matrix
        X_raw = self.df[self.candidate_cols].copy().astype(float)
        X_filled = X_raw.fillna(X_raw.median())
        
        # Standardize
        self.scaler = StandardScaler().fit(X_filled)
        X_scaled = self.scaler.transform(X_filled)

        # PCA
        self.pca = PCA(n_components=min(self.N_COMPONENTS_PCA, X_scaled.shape[1]))
        X_pca = self.pca.fit_transform(X_scaled)

        # Fit GMM
        n_samples = X_scaled.shape[0]
        if n_samples < self.GMM_COMPONENTS:
            logger.warning(f"Skipping {self.current_shot_name}: Not enough samples ({n_samples})")
            return False

        self.gmm = GaussianMixture(
            n_components=self.GMM_COMPONENTS, 
            covariance_type='full', 
            random_state=self.RANDOM_STATE, 
            n_init=5
        )
        self.gmm.fit(X_scaled)
        
        # Assign labels and confidence
        probs = self.gmm.predict_proba(X_scaled)
        self.df['gmm_label'] = self.gmm.predict(X_scaled)
        self.df['gmm_confidence'] = probs.max(axis=1)
        
        return True

    def cluster_ordering_score_matrix(self, df_local):
        """Calculates proxy score for ranking clusters."""
        core_power = [c for c in self.candidate_cols if "__rms" in c or "__iEMG" in c]
        core_freq = [c for c in self.candidate_cols if "__mnf" in c or "__mdf" in c]
        core_timing = [c for c in self.candidate_cols if "__time_to_peak" in c or "__dur_halfmax" in c]
        
        score = pd.Series(0.0, index=df_local.index)
        if core_power:
            score = score.add(zscore(df_local[core_power].mean(axis=1).fillna(0.0)), fill_value=0)
        if core_freq:
            score = score.add(0.5 * zscore(df_local[core_freq].mean(axis=1).fillna(0.0)), fill_value=0)
        if core_timing:
            score = score.add(-0.7 * zscore(df_local[core_timing].mean(axis=1).fillna(0.0)), fill_value=0)
        return score

    def save(self):
        """Saves labeled data and models into shot-specific folders."""
        output_shot_dir = self.cluster_dir / self.current_shot_name
        output_shot_dir.mkdir(parents=True, exist_ok=True)
        
        # Rank clusters by performance
        self.df['_proxy_score'] = self.cluster_ordering_score_matrix(self.df)
        cluster_means = self.df.groupby('gmm_label')['_proxy_score'].mean().sort_values(ascending=True)
        cluster_order = list(cluster_means.index)

        performance_labels = ["Poor", "Below-Average", "Average", "Excellent"]
        label_map = {cluster_idx: performance_labels[i] for i, cluster_idx in enumerate(cluster_order)}

        self.df['performance_level'] = self.df['gmm_label'].map(label_map)
        
        # Save Labeled CSV
        out_csv = output_shot_dir / "features_master_labeled_gmm4.csv"
        self.df.to_csv(out_csv, index=False)

        # Save Models
        pd.to_pickle(self.gmm, output_shot_dir / "gmm_model.pkl")
        pd.to_pickle(self.scaler, output_shot_dir / "scaler.pkl")
        pd.to_pickle(self.pca, output_shot_dir / "pca.pkl")
        
        logger.info(f"Successfully saved all outputs to {output_shot_dir}")

    def run(self):
        """Orchestrates the pipeline across all detected shot folders."""
        for shot_folder in self.shot_paths:
            try:
                success = self.fit_GMM(shot_folder)
                if success:
                    self.save()
            except Exception as e:
                logger.error(f"Error in {shot_folder.name}: {e}")
                raise CustomException(e, sys)

if __name__ == "__main__":
    standardization_clustering = Standardize_Cluster(TRAINING_FEATURES_DIR, TRAINING_CLUSTERS_DIR)
    standardization_clustering.run()