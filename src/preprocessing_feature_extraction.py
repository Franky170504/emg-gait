import os
import glob
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.integrate import trapezoid

class EMGPipeline:
    def __init__(self, input_folder, output_folder, fs=2000.0, mvc_value=1.0):
        """
        Initialize the EMG Processing Pipeline.
        
        Args:
            input_folder (str): Path to the folder containing raw CSV files.
            output_folder (str): Path to save processed files and features.
            fs (float): Sampling frequency in Hz (default 2000.0).
            mvc_value (float): Maximum Voluntary Contraction value for normalization (default 1.0).
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.fs = fs
        self.mvc_value = mvc_value
        
        # Ensure output directory exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print(f"Created output directory: {self.output_folder}")

    def _process_signal(self, raw_signal):
        """
        Applies filters, rectification, and envelope smoothing.
        """
        # Safety Flatten
        raw_signal = np.asarray(raw_signal).flatten()
        
        # 1. Bandpass Filter (20-450 Hz)
        sos_bp = signal.butter(4, [20, 450], btype='bandpass', fs=self.fs, output='sos')
        emg_bandpassed = signal.sosfiltfilt(sos_bp, raw_signal)
        
        # 2. Notch Filter (50 Hz)
        b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=self.fs)
        emg_cleaned = signal.filtfilt(b_notch, a_notch, emg_bandpassed)
        
        # 3. Rectification
        emg_rectified = np.abs(emg_cleaned)
        
        # 4. Envelope (RMS Rolling Window 50ms)
        window_ms = 50 
        window_samples = int((window_ms / 1000) * self.fs)
        
        emg_envelope = pd.Series(emg_rectified).rolling(
            window=window_samples, center=True, min_periods=1
        ).apply(lambda x: np.sqrt(np.mean(x**2)))
        
        return emg_envelope.values

    def _extract_features(self, df_processed):
        """
        Extracts statistical features from the processed envelope dataframe.
        """
        features = {}
        time_col = df_processed['Time'].values
        
        # Identify muscle columns (exclude Time)
        muscle_cols = [col for col in df_processed.columns if col != 'Time']
        
        for muscle in muscle_cols:
            envelope = df_processed[muscle].values
            
            # A. Peak Amplitude
            peak_amp = np.max(envelope)
            
            # B. Mean Amplitude
            mean_amp = np.mean(envelope)
            
            # C. Time to Peak
            peak_idx = np.argmax(envelope)
            time_to_peak = time_col[peak_idx]
            
            # D. Integrated EMG (iEMG)
            iemg = trapezoid(envelope, dx=1/self.fs)
            
            # E. Impact Phase (0.0s to 1.5s)
            end_index = int(1.5 * self.fs)
            if end_index > len(envelope): 
                end_index = len(envelope)
            impact_mean = np.mean(envelope[0:end_index])
            
            # Store features
            features[f"{muscle}_Max"] = peak_amp
            features[f"{muscle}_Mean"] = mean_amp
            features[f"{muscle}_TimePeak"] = time_to_peak
            features[f"{muscle}_iEMG"] = iemg
            features[f"{muscle}_ImpactMean"] = impact_mean
            
        return pd.DataFrame([features])

    def run(self):
        """
        Main execution function: iterates through input folder, processes files,
        and saves outputs.
        """
        # Find all CSV files
        csv_files = glob.glob(os.path.join(self.input_folder, "*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {self.input_folder}")
            return

        print(f"Found {len(csv_files)} files. Starting pipeline...")

        for file_path in csv_files:
            try:
                # 1. Load Data
                filename = os.path.basename(file_path)
                print(f"Processing: {filename}...")
                
                # Check if first column is index or data
                raw_df = pd.read_csv(file_path)
                
                # Clean up index column if it exists (common in pandas exports)
                if "Unnamed: 0" in raw_df.columns:
                    raw_df = raw_df.drop(columns=["Unnamed: 0"])
                if "index" in raw_df.columns:
                    raw_df = raw_df.drop(columns=["index"])

                # Handle Time column generation if missing or use existing
                if 'Time' in raw_df.columns:
                    t = raw_df['Time'].values
                else:
                    total_samples = len(raw_df)
                    t = np.linspace(0, total_samples / self.fs, total_samples)
                
                # Prepare storage for processed data
                processed_data_storage = {'Time': t}
                
                # Identify Muscle columns (exclude Time)
                muscle_cols = [c for c in raw_df.columns if c != 'Time']

                # 2. Process Signals per Muscle
                for muscle in muscle_cols:
                    raw_signal = raw_df[muscle].values
                    
                    # Apply filters and envelope
                    envelope = self._process_signal(raw_signal)
                    
                    # Normalize (% MVC)
                    norm_envelope = (envelope / self.mvc_value) * 100
                    
                    processed_data_storage[muscle] = norm_envelope

                df_processed = pd.DataFrame(processed_data_storage)

                # 3. Extract Features
                df_features = self._extract_features(df_processed)
                
                # Add filename reference to feature row for tracking
                df_features.insert(0, 'SourceFile', filename)

                # 4. Save Outputs
                base_name = os.path.splitext(filename)[0]
                
                # Save Time Series
                proc_path = os.path.join(self.output_folder, f"processed_{base_name}.csv")
                df_processed.to_csv(proc_path, index=False)
                
                # Save Features
                feat_path = os.path.join(self.output_folder, f"features_{base_name}.csv")
                df_features.to_csv(feat_path, index=False)
                
                print(f" -> Saved processed data and features for {filename}")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        print("\nPipeline execution finished.")

# ==========================================
# HOW TO RUN
# ==========================================
if __name__ == "__main__":
    # Define your folders here
    INPUT_DIR = r"Notebooks\data\processed"       # Folder where your raw CSVs are
    OUTPUT_DIR = r"data/processed_output" # Folder where you want results
    
    # Create the pipeline instance
    pipeline = EMGPipeline(
        input_folder=INPUT_DIR, 
        output_folder=OUTPUT_DIR, 
        fs=2000.0,       # Sampling frequency
        mvc_value=1.0    # Normalization factor
    )
    
    # Run the pipeline
    pipeline.run()