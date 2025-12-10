import os
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.integrate import trapezoid

class EMGPipeline:
    def __init__(self, input_folder, output_folder, fs=2000.0, mvc_value=1.0):
        """
        Initialize the Recursive EMG Processing Pipeline.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.fs = fs
        self.mvc_value = mvc_value

    def _load_and_format_data(self, file_path):
        """
        Loads CSV using specific headers [3, 4], cleans column names,
        and extracts 'EMG 1 (mV)' channels.
        """
        try:
            # 1. Read CSV with specific header rows
            df = pd.read_csv(file_path, header=[3, 4])
            
            # 2. Clean Column Names (Handle multi-level headers)
            # Row 3 (Index 0) is typically Sensor Name
            # Row 4 (Index 1) is typically Signal Type
            df.columns = pd.MultiIndex.from_tuples(
                [(c[0].split('(')[0].strip(), c[1].strip()) for c in df.columns]
            )
            
            # 3. Extract Time Column (Assumes it's the first column)
            time_col = df.iloc[:, 0] 
            
            # 4. Extract EMG Data
            # Note: We rely on the second level being 'EMG 1 (mV)'
            try:
                emg_data = df.xs('EMG 1 (mV)', level=1, axis=1).copy()
            except KeyError:
                print(f"   [!] Skipping {os.path.basename(file_path)}: 'EMG 1 (mV)' columns not found.")
                return None

            # 5. Insert Time back into the dataframe at the start
            emg_data.insert(0, 'Time', time_col)
            
            # 6. Reorder Columns
            target_order = [
                'Time', 
                'Rectus Femoris right', 'Rectus Femoris left', 
                'Hamstrings right', 'Hamstrings left', 
                'TibilaisÂ Anterior right', 'TibilaisÂ Anterior left', 
                'Gastrocnemius right', 'Gastrocnemius left'
            ]
            
            # Reindex aligns data to target_order. 
            emg_data = emg_data.reindex(columns=target_order)
            
            # Drop empty columns if any
            emg_data = emg_data.dropna(axis=1, how='all')
            
            return emg_data

        except Exception as e:
            print(f"   [!] Error loading {os.path.basename(file_path)}: {e}")
            return None

    def _process_signal(self, raw_signal):
        """ Applies filters, rectification, and envelope smoothing. """
        if np.all(np.isnan(raw_signal)):
            return raw_signal
            
        raw_signal = np.asarray(raw_signal).flatten()
        
        # Bandpass
        sos_bp = signal.butter(4, [20, 450], btype='bandpass', fs=self.fs, output='sos')
        emg_bandpassed = signal.sosfiltfilt(sos_bp, raw_signal)
        
        # Notch
        b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=self.fs)
        emg_cleaned = signal.filtfilt(b_notch, a_notch, emg_bandpassed)
        
        # Rectify & Envelope
        emg_rectified = np.abs(emg_cleaned)
        window_ms = 50 
        window_samples = int((window_ms / 1000) * self.fs)
        
        emg_envelope = pd.Series(emg_rectified).rolling(
            window=window_samples, center=True, min_periods=1
        ).apply(lambda x: np.sqrt(np.mean(x**2)))
        
        return emg_envelope.values

    def _extract_features(self, df_processed):
        """ Extracts statistical features. """
        features = {}
        time_col = df_processed['Time'].values
        
        muscle_cols = [col for col in df_processed.columns if col != 'Time']
        
        for muscle in muscle_cols:
            envelope = df_processed[muscle].values
            
            if np.all(np.isnan(envelope)):
                continue

            peak_amp = np.max(envelope)
            mean_amp = np.mean(envelope)
            peak_idx = np.argmax(envelope)
            time_to_peak = time_col[peak_idx]
            iemg = trapezoid(envelope, dx=1/self.fs)
            
            end_index = int(1.5 * self.fs)
            if end_index > len(envelope): end_index = len(envelope)
            impact_mean = np.mean(envelope[0:end_index])
            
            features[f"{muscle}_Max"] = peak_amp
            features[f"{muscle}_Mean"] = mean_amp
            features[f"{muscle}_TimePeak"] = time_to_peak
            features[f"{muscle}_iEMG"] = iemg
            features[f"{muscle}_ImpactMean"] = impact_mean
            
        return pd.DataFrame([features])

    def run(self):
        """
        Walks through input folder recursively and replicates structure in output.
        """
        print(f"Scanning '{self.input_folder}' for CSV files...")
        files_processed_count = 0

        for root, dirs, files in os.walk(self.input_folder):
            for file in files:
                if file.endswith(".csv"):
                    
                    full_input_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, self.input_folder)
                    
                    target_dir = os.path.join(self.output_folder, relative_path)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)

                    try:
                        print(f"Processing: {file}...", end=" ")
                        
                        # --- 1. LOAD DATA ---
                        raw_df = self._load_and_format_data(full_input_path)
                        
                        if raw_df is None or raw_df.empty:
                            continue

                        # Prepare storage with Time column
                        processed_data_storage = {'Time': raw_df['Time'].values}
                        
                        # Identify Muscle Columns (All non-Time columns)
                        muscle_cols = [c for c in raw_df.columns if c != 'Time']

                        # --- 2. PROCESS BY NAME (Fixes Shift Issue) ---
                        # Instead of iterating by index (i), we iterate by muscle name directly.
                        # This ensures the data maps correctly even if order varies.
                        for muscle_name in muscle_cols:
                            
                            # Get data by name
                            raw_data = raw_df[muscle_name].values
                            
                            # Process
                            envelope = self._process_signal(raw_data)
                            
                            # Normalize
                            norm_envelope = (envelope / self.mvc_value) * 100
                            
                            # Store by name
                            processed_data_storage[muscle_name] = norm_envelope

                        # Create Output DataFrame
                        df_processed = pd.DataFrame(processed_data_storage)

                        # --- 3. FEATURE EXTRACTION ---
                        df_features = self._extract_features(df_processed)
                        df_features.insert(0, 'SourceFile', file)

                        # --- 4. SAVE ---
                        base_name = os.path.splitext(file)[0]
                        proc_filename = f"processed_{base_name}.csv"
                        feat_filename = f"features_{base_name}.csv"
                        
                        df_processed.to_csv(os.path.join(target_dir, proc_filename), index=False)
                        df_features.to_csv(os.path.join(target_dir, feat_filename), index=False)
                        
                        print("Done.")
                        files_processed_count += 1

                    except Exception as e:
                        print(f"\n   Error processing {full_input_path}: {e}")

        print(f"\nPipeline finished. Processed {files_processed_count} files.")

if __name__ == "__main__":
    INPUT_ROOT = r"artifacts/raw"       
    OUTPUT_ROOT = r"data/processed_output" 
    
    pipeline = EMGPipeline(INPUT_ROOT, OUTPUT_ROOT)
    pipeline.run()