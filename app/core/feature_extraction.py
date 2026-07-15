import sys
import numpy as np
import pandas as pd

from scipy.signal import welch
from scipy.ndimage import uniform_filter1d
from scipy.integrate import trapezoid
from src.custom_exception import CustomException

class FeatureExtraction:
    def __init__(self):
        self.FS=2148.1481
        self.WELCH_NPERSEG=512
        self.RMS_WINDOW=(
            int(0.05*self.FS)
        )

    def moving_rms(self,x):
        return np.sqrt(
            uniform_filter1d(
                x*x,
                size=self.RMS_WINDOW,
                mode="nearest"
            )
        )

    def time_features(self,x):
        return {
            "mean":np.mean(x),
            "std":np.std(x),
            "rms":np.sqrt(np.mean(x*x)),
            "mav":np.mean(abs(x)),
            "wl":np.sum(abs(np.diff(x))),
            "peak":np.max(x),
            "iEMG":trapezoid(abs(x))
        }

    def frequency_features(self,x):
        f,p=welch(
            x,
            fs=self.FS,
            nperseg=min(
                len(x),
                self.WELCH_NPERSEG
            )
        )
        total=np.sum(p)+1e-12
        mnf=np.sum(f*p)/total
        idx=np.searchsorted(
            np.cumsum(p),
            total/2
        )
        return {
            "mnf":mnf,
            "mdf":f[min(idx,len(f)-1)]
        }

    def process_dataframe(self,filtered_df):
        try:
            df=filtered_df.copy()
            time_cols=[
                c for c in df.columns
                if "time" in c.lower()
            ]
            if time_cols:
                df=df.drop(
                    columns=time_cols[0]
                )
            output={}
            peaks={}
            channel_features={}
            for col in df.columns:
                x=pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).fillna(0).values
                feat={}
                feat.update(
                    self.time_features(x)
                )
                feat.update(
                    self.frequency_features(x)
                )
                rms=self.moving_rms(x)
                feat["mrms_mean"]=np.mean(rms)
                feat["mrms_peak"]=np.max(rms)
                channel_features[col]=feat
                peaks[col]=feat["peak"]
            trial_peak=max(peaks.values())
            if trial_peak==0:
                trial_peak=1
            for ch,features in channel_features.items():
                for k,v in features.items():
                    output[f"{ch}__{k}"]=v
                    if k in [
                        "rms",
                        "peak",
                        "mav",
                        "iEMG",
                        "mrms_mean",
                        "mrms_peak"
                    ]:
                        output[
                            f"{ch}__{k}_rel"
                        ]=v/trial_peak
            output["trial_peak_of_channel_peaks"]=trial_peak
            return pd.DataFrame([output])

        except Exception as e:
            raise CustomException(e,sys)