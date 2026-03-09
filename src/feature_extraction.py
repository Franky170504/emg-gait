import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm
from pathlib import Path

from src.custom_exception import CustomException
from src.logger import get_logger
from config.path_config import *

logger = get_logger(__name__)

class FeatureExtraction:
    def __init__(self, filtered_dir, feature_dir):
        self.filtered_dir = Path(filtered_dir)
        self.feature_dir = Path(feature_dir)  
        self.FS = 2148.1481
        self.WELCH_NPERSEG = 512
        self.MIN_NUM_SAMPLES = 50
        self.CANONICAL_CHANNEL_ORDER = [
            'Rectus Femoris right', 'Rectus Femoris left', 
            'Hamstrings right', 'Hamstrings left', 
            'TibilaisÂ Anterior right', 'TibilaisÂ Anterior left', 
            'Gastrocnemius right', 'Gastrocnemius left'
        ]
        # RMS window (seconds) for moving-RMS
        self.RMS_WINDOW_MS = 50
        self.RMS_WINDOW_SAMPLES = max(1, int((self.RMS_WINDOW_MS/1000.0) * self.FS))

    def safe_welch(self, x, fs, nperseg):
        nperseg_eff = min(len(x), max(16, nperseg))
        try:
            f, Pxx = welch(x, fs=fs, nperseg=nperseg_eff)
        except Exception:
            f = np.array([0.0])
            Pxx = np.array([0.0])
        return f, Pxx

    def moving_rms(self, x, window_samples):
        if len(x) < window_samples or window_samples <= 1:
            return np.sqrt(np.mean(x**2)) * np.ones_like(x)
        sq = x.astype(float)**2
        mean_sq = uniform_filter1d(sq, size=window_samples, mode='nearest')
        return np.sqrt(mean_sq)

    def canonicalize_emg_df(self, df, canonical_order):
        cols_present = df.columns.tolist()
        ordered = []
        for c in canonical_order:
            if c in cols_present:
                ordered.append(c)
            else:
                df[c] = np.nan
                ordered.append(c)
        remaining = [c for c in cols_present if c not in canonical_order]
        ordered += remaining
        return df[ordered]
    
    def extract_time_features(self, x):
        x = np.asarray(x).astype(float)
        if x.size == 0:
            return {
                "mean": np.nan, "std": np.nan, "rms": np.nan,
                "mav": np.nan, "wl": np.nan, "peak": np.nan, "iEMG": np.nan
            }
        return {
            "mean": float(np.mean(x)),
            "std":  float(np.std(x)),
            "rms":  float(np.sqrt(np.mean(x**2))),
            "mav":  float(np.mean(np.abs(x))),
            "wl":   float(np.sum(np.abs(np.diff(x)))),
            "peak": float(np.max(x)),
            "iEMG": float(np.trapz(np.abs(x)))  # numeric integral
        }

    def extract_freq_features(self, x):
        x = np.asarray(x).astype(float)
        if len(x) < 4:
            return {"mnf": np.nan, "mdf": np.nan, "bp_20_60": np.nan, "bp_60_100": np.nan, "bp_100_200": np.nan}
        f, Pxx = self.safe_welch(x, self.FS, self.WELCH_NPERSEG)
        total = np.sum(Pxx) + 1e-12
        mnf = float(np.sum(f * Pxx) / total)
        csum = np.cumsum(Pxx)
        half = total / 2.0
        idx = np.searchsorted(csum, half)
        mdf = float(f[idx]) if idx < len(f) else float(f[-1])
        def bandpow(a,b):
            mask = (f >= a) & (f <= b)
            return float(np.trapz(Pxx[mask], f[mask])) if np.any(mask) else 0.0
        return {"mnf": mnf, "mdf": mdf, "bp_20_60": bandpow(20,60), "bp_60_100": bandpow(60,100), "bp_100_200": bandpow(100,200)}
    
    def process_file_to_features(self, path):
        """Processes a single file and returns a dictionary of features."""
        df = pd.read_csv(path)
        df = self.canonicalize_emg_df(df, self.CANONICAL_CHANNEL_ORDER) 
        
        # NEW: Determine Shot Type from the folder structure
        # This assumes path is: data/filtered/shot_type/player_1.csv
        relative_path = Path(path).relative_to(self.filtered_dir)
        shot_type = relative_path.parts[0] # Extracts 'chipshot'
        
        time_cols = [c for c in df.columns if 'time' in c.lower() or 'timestamp' in c.lower()]
        time_col = time_cols[0] if time_cols else None
        emg_cols = [c for c in df.columns if c != time_col]
        n_samples = df.shape[0]

        row = {
            'file': str(path),
            'shot_type': shot_type, # NEW: Added to row for grouping
            'n_samples': n_samples,
            'fs_used': self.FS
        }

        # Extract Player Name from filename (e.g., 'jordan' from 'jordan_1.csv')
        stem = Path(path).stem
        row['player'] = stem.split("_")[0] if "_" in stem else shot_type
        per_channel_data = {}
        channel_means = {}
        channel_peaks = {}

        for ch in emg_cols:
            x = pd.to_numeric(df[ch], errors='coerce').fillna(0).values.astype(float)
            per_channel_data[ch] = {}
            per_channel_data[ch].update(self.extract_time_features(x))
            per_channel_data[ch].update(self.extract_freq_features(x))
            
            mr = self.moving_rms(x, self.RMS_WINDOW_SAMPLES)
            per_channel_data[ch]['mrms_mean'] = float(np.mean(mr))
            per_channel_data[ch]['mrms_peak'] = float(np.max(mr))
            
            channel_means[ch] = per_channel_data[ch].get('mean', np.nan)
            channel_peaks[ch] = per_channel_data[ch].get('peak', np.nan)

        trial_peak = np.nanmax(list(channel_peaks.values())) if channel_peaks else 1.0
        if trial_peak == 0: trial_peak = 1.0

        for ch in emg_cols:
            for k, v in per_channel_data[ch].items():
                row[f"{ch}__{k}"] = v
                if k in ('rms','peak','iEMG','mrms_mean','mrms_peak','mav'):
                    row[f"{ch}__{k}_rel"] = v / trial_peak

        row['trial_peak_of_channel_peaks'] = float(trial_peak)
        return row
        

    def find_csv_files(self, root_dir):
        try:
            fl = []
            for root, _, files in os.walk(root_dir):
                for f in files:
                    if f.lower().endswith(".csv"):
                        fl.append(os.path.join(root, f))
            return sorted(fl)
        except Exception as e:
            logger.error(f"Failed processing: {e}")
    
    def run(self):
        try:
            files = sorted(list(self.filtered_dir.rglob('*.csv')))
            logger.info(f"Found {len(files)} files for feature extraction.")
            
            all_features = []
            
            # Step 1: Extract features from all files and build a master list
            for p in tqdm(files, desc="Extracting Features"):
                try:
                    feat_row = self.process_file_to_features(p)
                    if feat_row:
                        all_features.append(feat_row)
                except Exception as e:
                    logger.error(f"Failed processing {p}: {e}")

            if not all_features:
                logger.warning("No features extracted.")
                return

            # Convert to master DataFrame
            df_master = pd.DataFrame(all_features)

            # Step 2: Loop through unique shot types to create separate sub-folders
            shot_types = df_master['shot_type'].unique()
            
            for shot in shot_types:
                # Filter data for this specific shot type
                shot_df = df_master[df_master['shot_type'] == shot].copy()
                shot_dir = self.feature_dir / shot
                shot_dir.mkdir(parents=True, exist_ok=True) # Create 'data/features/chipshot/'

                # Save CSV specific to this shot type
                shot_df.to_csv(shot_dir / "features_master.csv", index=False)


                # Save the segmented Numpy files and column names
                logger.info(f"Successfully exported {shot} features to {shot_dir}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    feature_extraction = FeatureExtraction(TRAINING_FILTERED_DIR, TRAINING_FEATURES_DIR)
    feature_extraction.run()