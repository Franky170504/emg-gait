import sys
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)


class DataCleaning:
    def __init__(self):
        self.target_order = [
            'Time', 
            'Rectus Femoris right', 'Rectus Femoris left', 
            'Hamstrings right', 'Hamstrings left', 
            'TibilaisÂ Anterior right', 'TibilaisÂ Anterior left', 
            'Gastrocnemius right', 'Gastrocnemius left'
        ]

    def process_file(self, uploaded_file):
        """
        uploaded_file : Streamlit UploadedFile object
        Returns cleaned dataframe
        """

        try:
            df = pd.read_csv(uploaded_file, header=[3, 4])

            # Remove empty columns
            df = df.dropna(axis=1, how="all")

            # Clean MultiIndex column names
            df.columns = pd.MultiIndex.from_tuples(
                [(c[0].split('(')[0].strip(), c[1].strip()) for c in df.columns]
            )

            time_col = df.iloc[:, 0]
            try:
                emg_data = df.xs('EMG 1 (mV)', level=1, axis=1).copy()
            except KeyError:
                
                return None

            emg_data.insert(0, 'Time', time_col)
            emg_data = emg_data.reindex(columns=self.target_order)
            
            return emg_data

        except Exception as e:
            raise CustomException(e, sys)