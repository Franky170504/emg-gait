import os

## Data Ingestion ##

RAW_DIR = r"artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR,"raw.csv")
# TRAIN_FILE_PATH = os.path.join(RAW_DIR,"train.csv")
# TEST_FILE_PATH = os.path.join(RAW_DIR,"test.csv")

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.yaml")

## Data Processing ##

PROCESSED_DIR = r"artifacts/processed"
PROCESSED_TRAIN_FILE_PATH = os.path.join(PROCESSED_DIR,"train.csv")
PROCESSED_TEST_FILE_PATH = os.path.join(PROCESSED_DIR,"test.csv")

## Model Path ##

MODEL_OUTPUT_PATH = r"artifacts/models/lightgbm.pkl"
# MODEL_OUTPUT_PATH = os.path.join(MODEL_OUTPUT_DIR,"")