from app.config.path_config import *
from logic.src.data_cleaning import DataCleaning
from logic.src.data_filtering import Filtering
from logic.src.feature_extraction import FeatureExtraction
from logic.src.inference import Inference

if __name__ == "__main__":

    shot_name = input(
        "Enter shot type "
        "(chipshot / curve / longpass / powershot / "
        "shortpass / trivela / volley): "
    )

    player_name = input("Enter Playername: ")

    data_processor = DataCleaning(INFERENCE_RAW_DIR, INFERENCE_CLEANED_DIR)
    data_processor.run()

    data_filter = Filtering(INFERENCE_CLEANED_DIR, INFERENCE_FILTERED_DIR)
    data_filter.run()

    feature_extraction = FeatureExtraction(INFERENCE_FILTERED_DIR, INFERENCE_FEATURES_DIR)
    feature_extraction.run()

    inference = Inference(
    shot_name=shot_name,
    feature_root_dir="artifacts/inference/features",
    player_name=player_name,
    model_name="gradient_boost"
    )
    inference.run()

