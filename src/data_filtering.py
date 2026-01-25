import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from src.custom_exception import CustomException
from src.logger import get_logger
from config.path_config import *

logger = get_logger(__name__)

class Filtering:
    def __init__(self, processed_dir, filtered_dir):
        self.processed_dir = Path(processed_dir)
        self.filtered_dir = Path(filtered_dir)
        self.FS = 2148.1481            
        self.NOTCH_FREQ = 50.0
        self.NOTCH_Q = 30
        self.LOWCUT = 20.0
        self.HIGHCUT = 450.0
        self.ENV_CUTOFF = 6.0          
        self.FILTER_ORDER = 4

    def notch_filter(self, x, freq, fs, Q):
        try:
            if len(x) < 4:
                return x
            w0 = freq / (0.5 * fs)
            b, a = signal.iirnotch(w0, Q)
            return signal.filtfilt(b, a, x)
        
        except CustomException as e:
            logger.error(f"Error while applying notch filter {e}")

    def bandpass_filter(self, x, lowcut, highcut, fs, order):
        try:
            if len(x) < (order*3):
                return x
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            if not (0 < low < high < 1):
                high = min(0.99, high)
            b, a = signal.butter(order, [low, high], btype='band')
            return signal.filtfilt(b, a, x)
        
        except CustomException as e:
            logger.error(f"Error while applying bandpass filter {e}")

    def lowpass_filter(self, x, cutoff, fs, order):
        try:
            if len(x) < (order*3):
                return x
            nyq = 0.5 * fs
            c = cutoff / nyq
            c = min(0.99, c)
            b, a = signal.butter(order, c, btype='low')
            return signal.filtfilt(b, a, x)
        
        except CustomException as e:
            logger.error(f"Error while applying lowpass filter {e}")
    
    def find_csv_files(self, root_dir):
        try: 
            paths = []
            for root, _, files in os.walk(root_dir):
                for f in files:
                    if f.lower().endswith(".csv"):
                        paths.append(os.path.join(root, f))
            return sorted(paths)
        
        except CustomException as e:
            logger.error(f"Error while finding csv {e}")

    def detect_time_and_emg_columns(self, df):
        try:
            time_candidates = [c for c in df.columns if 'time' in str(c).lower() or 'timestamp' in str(c).lower()]
            time_col = time_candidates[0] if time_candidates else None
            numeric = []
            for c in df.columns:
                if c == time_col:
                    continue
                series = pd.to_numeric(df[c], errors='coerce')
                if series.notnull().sum() > 5:
                    numeric.append(c)
            return time_col, numeric
        
        except CustomException as e:
            logger.error(f"Error while detecting time column {e}")
    
    def preprocess_and_save(self, src_path, dst_path):
        try:
            df = pd.read_csv(src_path)
            time_col, emg_cols = self.detect_time_and_emg_columns(df)
            if len(emg_cols) == 0:
                logger.warning("Skipping (no EMG columns found):", src_path)
                return

            for c in emg_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(float)

            if time_col and time_col in df.columns:
                time_vec = pd.to_numeric(df[time_col], errors='coerce').values
            else:
                time_vec = None

            emg = df[emg_cols].values.astype(float)

            env_mat = np.zeros_like(emg)
            for i in range(emg.shape[1]):
                x = emg[:, i]
                try:
                    x = self.notch_filter(x, self.NOTCH_FREQ, self.FS, self.NOTCH_Q)
                except CustomException:
                    pass
                try:
                    x = self.bandpass_filter(x, self.LOWCUT, self.HIGHCUT, self.FS, self.FILTER_ORDER)
                except CustomException:
                    pass
                x_rect = np.abs(x)
                try:
                    x_env = self.lowpass_filter(x_rect, self.ENV_CUTOFF, self.FS, self.FILTER_ORDER)
                except CustomException:
                    x_env = x_rect
                env_mat[:, i] = x_env

            out_df = pd.DataFrame(env_mat, columns=emg_cols)
            if time_vec is not None and len(time_vec) == out_df.shape[0]:
                out_df.insert(0, time_col, time_vec)
            else:
                
                n = out_df.shape[0]
                out_df.insert(0, "time", np.arange(n) / self.FS)

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            out_df.to_csv(dst_path, index=False)
        
        except CustomException as e:
            logger.error(f"Error while filtering {e}")

    def run(self):
        try:
            src_files = self.find_csv_files(self.processed_dir)
            logger.info(f"Staring with data filtering")            

            for src in tqdm(src_files, desc="Filtering Data"):
                rel = os.path.relpath(src, self.processed_dir)
                dst = os.path.join(self.filtered_dir, rel)
                try:
                    self.preprocess_and_save(src, dst)
                    logger.info(f"Filtered {len(src_files)} CSV files.")

                except CustomException as e:
                    logger.error(f"Error processing {src}: {e}")

        except CustomException as e:
            logger.error(f"Error while running the pipeline {e}")

if __name__ == "__main__":
    data_filter = Filtering(TRAINING_CLEANED_DIR, TRAINING_FILTERED_DIR)
    data_filter.run()
