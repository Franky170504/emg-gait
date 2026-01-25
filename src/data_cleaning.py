import sys
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import *

logger = get_logger(__name__)

class DataCleaning:
    def __init__(self, raw_dir, processed_dir):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.target_order = [
            'Time', 
            'Rectus Femoris right', 'Rectus Femoris left', 
            'Hamstrings right', 'Hamstrings left', 
            'TibilaisÂ Anterior right', 'TibilaisÂ Anterior left', 
            'Gastrocnemius right', 'Gastrocnemius left'
        ]

    def process_single_file(self, file_path):
        try:
            df = pd.read_csv(file_path, header=[3, 4],)
            df.dropna(axis=1)

            df.columns = pd.MultiIndex.from_tuples(
                [(c[0].split('(')[0].strip(), c[1].strip()) for c in df.columns]
            )

            time_col = df.iloc[:, 0]
            try:
                emg_data = df.xs('EMG 1 (mV)', level=1, axis=1).copy()
            except KeyError:
                logger.warning(f"'EMG 1 (mV)' not found in {file_path}. Skipping.")
                return None

            emg_data.insert(0, 'Time', time_col)
            emg_data = emg_data.reindex(columns=self.target_order)
            
            return emg_data

        except Exception as e:
            raise CustomException(e, sys)

    def run_cleaning_pipeline(self):
        processed_count = 0
        
        try:
            logger.info(f"Data cleaning started")
            files = list(self.raw_dir.rglob('*.csv'))
            for file_path in tqdm(files, desc="Cleaning Files"):
                parts = file_path.relative_to(self.raw_dir).parts
                
                if len(parts) >= 3:
                    shot_type = parts[0]
                    subject = parts[1]
                    filename = parts[2]
                    
                    shot_dir = self.processed_dir / shot_type
                    shot_dir.mkdir(parents=True, exist_ok=True)
                    
                    new_filename = f"{subject}_{filename}"
                    save_path = shot_dir / new_filename
                    
                    cleaned_df = self.process_single_file(file_path)
                    
                    if cleaned_df is not None:
                        cleaned_df.to_csv(save_path, index=False)
                        processed_count += 1
                        logger.info(f"Saved: {save_path}")
                else:
                    logger.warning(f"File structure for {file_path} is too shallow. Skipping.")

                logger.info(f"{processed_count} files cleaned")

            return processed_count
        
        except CustomException as e:
            logger.error(f"Error in cleaning the data {e}")

    def run(self):
        try:
            logger.info("Starting Shot-Type Based Data Processing...")
            count = self.run_cleaning_pipeline()
            logger.info(f"Successfully processed {count} files.")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_processor = DataCleaning(TRAINING_RAW_DIR, TRAINING_CLEANED_DIR)
    data_processor.run()