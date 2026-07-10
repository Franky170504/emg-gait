import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)


class Cleaning:
    def __init__(self):
        self.target_order = [
            'Time', 
            'Rectus Femoris right', 'Rectus Femoris left', 
            'Hamstrings right', 'Hamstrings left', 
            'TibilaisÂ Anterior right', 'TibilaisÂ Anterior left', 
            'Gastrocnemius right', 'Gastrocnemius left'
        ]

        self.muscle_colors = {
            'Rectus Femoris right': "#9eff1f", 
            'Rectus Femoris left': "#ff1fa9", 
            'Hamstrings right':"#ff8a00", 
            'Hamstrings left': "#00167a", 
            'TibilaisÂ Anterior right' : "#ff0000", 
            'TibilaisÂ Anterior left': "#5cd4d3", 
            'Gastrocnemius right' : "#ffd21f", 
            'Gastrocnemius left' :"#9440dd", 
        }

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
        
    
    def get_muscle_list(self, cleaned_df):
        """Returns all muscle names."""
        return [col for col in cleaned_df.columns if col != "Time"]

    def create_plot(self, cleaned_df, selected_muscles):

        fig = go.Figure()

        for muscle in selected_muscles:
            fig.add_trace(
                go.Scatter(
                    x=cleaned_df["Time"],
                    y=cleaned_df[muscle],
                    mode="lines+markers",
                    name=muscle,

                    line=dict(
                        color=self.muscle_colors[muscle],
                        width=3,
                        shape="spline",      # Smooth curve
                        smoothing=1.2
                    ),

                    marker=dict(
                        size=5,
                        color=self.muscle_colors[muscle],
                    ),

                    opacity=0.9
                )
            )

        fig.update_layout(

            title="EMG Signals",

            template="plotly_dark",

            height=650,

            hovermode="x unified",

            xaxis=dict(
                title="Time (s)",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.12)",
                zeroline=False
            ),

            yaxis=dict(
                title="EMG (mV)",
                showgrid=True,
                gridcolor="rgba(255,255,255,0.12)",
                zeroline=False
            ),

            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),

            font=dict(
                size=14,
                color="white"
            )
        )

        return fig