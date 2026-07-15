import sys
import numpy as np
import pandas as pd

from scipy.signal import welch
from scipy.ndimage import uniform_filter1d
from scipy.integrate import trapezoid

from src.custom_exception import CustomException
from src.logger import get_logger


logger = get_logger(__name__)


class FeatureExtraction:

    def __init__(self):

        self.FS = 2148.1481

        self.WELCH_NPERSEG = 512

        self.MIN_NUM_SAMPLES = 50

        self.CANONICAL_CHANNEL_ORDER = [

            'Rectus Femoris right',
            'Rectus Femoris left',

            'Hamstrings right',
            'Hamstrings left',

            'TibilaisÂ Anterior right',
            'TibilaisÂ Anterior left',

            'Gastrocnemius right',
            'Gastrocnemius left'
        ]


        self.RMS_WINDOW_MS = 50

        self.RMS_WINDOW_SAMPLES = max(
            1,
            int(
                (self.RMS_WINDOW_MS / 1000.0)
                *
                self.FS
            )
        )


    # ---------------------------------------------------
    # Frequency calculation
    # ---------------------------------------------------

    def safe_welch(self, x):

        nperseg_eff = min(
            len(x),
            max(16, self.WELCH_NPERSEG)
        )

        try:

            f, Pxx = welch(
                x,
                fs=self.FS,
                nperseg=nperseg_eff
            )

        except Exception:

            f = np.array([0.0])

            Pxx = np.array([0.0])


        return f, Pxx



    # ---------------------------------------------------
    # Moving RMS
    # ---------------------------------------------------

    def moving_rms(
            self,
            x,
            window_samples
    ):

        if (
            len(x) < window_samples
            or window_samples <= 1
        ):

            return (
                np.sqrt(
                    np.mean(x**2)
                )
                *
                np.ones_like(x)
            )


        sq = x.astype(float)**2


        mean_sq = uniform_filter1d(
            sq,
            size=window_samples,
            mode="nearest"
        )


        return np.sqrt(mean_sq)



    # ---------------------------------------------------
    # Channel ordering
    # ---------------------------------------------------

    def canonicalize_emg_df(
            self,
            df,
            canonical_order
    ):


        cols_present = df.columns.tolist()


        ordered = []


        for c in canonical_order:


            if c in cols_present:

                ordered.append(c)


            else:

                df[c] = np.nan

                ordered.append(c)



        remaining = [

            c for c in cols_present

            if c not in canonical_order

        ]


        ordered += remaining


        return df[ordered]



    # ---------------------------------------------------
    # Time domain features
    # ---------------------------------------------------

    def extract_time_features(
            self,
            x
    ):


        x = np.asarray(x).astype(float)


        if x.size == 0:

            return {

                "mean": np.nan,
                "std": np.nan,
                "rms": np.nan,
                "mav": np.nan,
                "wl": np.nan,
                "peak": np.nan,
                "iEMG": np.nan

            }


        return {


            "mean":
                float(np.mean(x)),


            "std":
                float(np.std(x)),


            "rms":
                float(
                    np.sqrt(
                        np.mean(x**2)
                    )
                ),


            "mav":
                float(
                    np.mean(
                        np.abs(x)
                    )
                ),


            "wl":
                float(
                    np.sum(
                        np.abs(
                            np.diff(x)
                        )
                    )
                ),


            "peak":
                float(
                    np.max(x)
                ),


            "iEMG":
                float(
                    trapezoid(
                        np.abs(x)
                    )
                )

        }



    # ---------------------------------------------------
    # Frequency domain features
    # ---------------------------------------------------

    def extract_freq_features(
            self,
            x
    ):


        x = np.asarray(x).astype(float)


        if len(x) < 4:

            return {

                "mnf": np.nan,
                "mdf": np.nan,

                "bp_20_60": np.nan,
                "bp_60_100": np.nan,
                "bp_100_200": np.nan

            }



        f, Pxx = self.safe_welch(x)


        total = np.sum(Pxx) + 1e-12



        mnf = (
            np.sum(
                f * Pxx
            )
            /
            total
        )



        csum = np.cumsum(Pxx)


        idx = np.searchsorted(
            csum,
            total / 2
        )


        mdf = (

            float(f[idx])

            if idx < len(f)

            else float(f[-1])

        )



        def bandpow(a,b):

            mask = (
                (f >= a)
                &
                (f <= b)
            )


            if np.any(mask):

                return float(
                    np.trapezoid(
                        Pxx[mask],
                        f[mask]
                    )
                )


            return 0.0



        return {


            "mnf": float(mnf),

            "mdf": float(mdf),


            "bp_20_60":
                bandpow(20,60),


            "bp_60_100":
                bandpow(60,100),


            "bp_100_200":
                bandpow(100,200)

        }



    # ---------------------------------------------------
    # MAIN FUNCTION USED BY STREAMLIT
    # ---------------------------------------------------

    def process_dataframe(
            self,
            filtered_df
    ):


        try:


            df = filtered_df.copy()



            df = self.canonicalize_emg_df(
                df,
                self.CANONICAL_CHANNEL_ORDER
            )



            time_cols = [

                c for c in df.columns

                if (
                    "time" in c.lower()

                    or

                    "timestamp" in c.lower()
                )

            ]



            time_col = (
                time_cols[0]
                if time_cols
                else None
            )



            emg_cols = [

                c for c in df.columns

                if c != time_col

            ]



            row = {


                "n_samples":
                    len(df),


                "fs_used":
                    self.FS

            }



            per_channel_data = {}

            channel_peaks = {}



            for ch in emg_cols:



                x = (

                    pd.to_numeric(
                        df[ch],
                        errors="coerce"
                    )

                    .fillna(0)

                    .values

                    .astype(float)

                )



                per_channel_data[ch] = {}



                per_channel_data[ch].update(

                    self.extract_time_features(x)

                )



                per_channel_data[ch].update(

                    self.extract_freq_features(x)

                )



                mr = self.moving_rms(

                    x,

                    self.RMS_WINDOW_SAMPLES

                )



                per_channel_data[ch][
                    "mrms_mean"
                ] = float(
                    np.mean(mr)
                )



                per_channel_data[ch][
                    "mrms_peak"
                ] = float(
                    np.max(mr)
                )



                channel_peaks[ch] = (

                    per_channel_data[ch]["peak"]

                )



            trial_peak = np.nanmax(

                list(channel_peaks.values())

            )



            if trial_peak == 0:

                trial_peak = 1.0



            for ch in emg_cols:



                for k,v in per_channel_data[ch].items():



                    row[
                        f"{ch}__{k}"
                    ] = v



                    if k in (

                        "rms",
                        "peak",
                        "iEMG",
                        "mrms_mean",
                        "mrms_peak",
                        "mav"

                    ):


                        row[
                            f"{ch}__{k}_rel"
                        ] = (

                            v / trial_peak

                        )



            row[
                "trial_peak_of_channel_peaks"
            ] = float(trial_peak)



            return pd.DataFrame([row])



        except Exception as e:


            raise CustomException(
                e,
                sys
            )