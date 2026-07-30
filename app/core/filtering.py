import sys
import numpy as np
import pandas as pd
from scipy import signal

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class Filtering:
    def __init__(self):
        self.FS = 2148.1481
        self.NOTCH_FREQ = 50
        self.NOTCH_Q = 30
        self.LOWCUT = 20
        self.HIGHCUT = 450
        self.ENV_CUTOFF = 6
        self.ORDER = 4

    def notch_filter(self, x):
        w0 = self.NOTCH_FREQ / (0.5*self.FS)
        b, a = signal.iirnotch(
            w0,
            self.NOTCH_Q
        )
        return signal.filtfilt(b,a,x)

    def bandpass_filter(self,x):
        nyq = self.FS/2
        low = self.LOWCUT/nyq
        high = self.HIGHCUT/nyq
        b,a = signal.butter(
            self.ORDER,
            [low,high],
            btype="band"
        )
        return signal.filtfilt(b,a,x)

    def lowpass_filter(self,x):
        cutoff = self.ENV_CUTOFF/(self.FS/2)
        b,a = signal.butter(
            self.ORDER,
            cutoff,
            btype="low"
        )
        return signal.filtfilt(b,a,x)

    def process_dataframe(self, cleaned_df):
        try:
            df = cleaned_df.copy()
            time_col = None
            for c in df.columns:
                if "time" in c.lower():
                    time_col=c
                    break

            emg_cols=[c for c in df.columns if c!=time_col]
            for col in emg_cols:
                x = pd.to_numeric(df[col],errors="coerce").fillna(0).values
                x=self.notch_filter(x)
                x=self.bandpass_filter(x)
                x=np.abs(x)
                x=self.lowpass_filter(x)
                df[col]=x
            return df

        except Exception as e:
            raise CustomException(e,sys)