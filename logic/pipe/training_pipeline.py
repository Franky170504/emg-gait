from app.config.path_config import *

from logic.src.data_cleaning import DataCleaning
from logic.src.data_filtering import Filtering
from logic.src.feature_extraction import FeatureExtraction
from logic.src.standardization_clustering import Standardize_Cluster
from logic.src.model_training import ModelTraining

if __name__ == "__main__":
    # data_processor = DataCleaning(TRAINING_RAW_DIR, TRAINING_CLEANED_DIR)
    # data_processor.run()

    # data_filter = Filtering(TRAINING_CLEANED_DIR, TRAINING_FILTERED_DIR)
    # data_filter.run()

    # feature_extraction = FeatureExtraction(TRAINING_FILTERED_DIR, TRAINING_FEATURES_DIR)
    # feature_extraction.run()

    # standardization_clustering = Standardize_Cluster(TRAINING_FEATURES_DIR, TRAINING_CLUSTERS_DIR)
    # standardization_clustering.run()

    trainer = ModelTraining(TRAINING_CLUSTERS_DIR)
    trainer.run()
