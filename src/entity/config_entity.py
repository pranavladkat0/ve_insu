import os
from dataclasses import dataclass
from datetime import datetime

from src.constants import *

TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# ========================
# PIPELINE CONFIG
# ========================
@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config = TrainingPipelineConfig()


# ========================
# DATA INGESTION
# ========================
class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            "data_ingestion"
        )

        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            "feature_store",
            "data.csv"
        )

        self.training_file_path = os.path.join(
            self.data_ingestion_dir,
            "ingested",
            "train.csv"
        )

        self.testing_file_path = os.path.join(
            self.data_ingestion_dir,
            "ingested",
            "test.csv"
        )

        self.train_test_split_ratio = 0.2
        self.collection_name = "vehicle_data"


# ========================
# DATA VALIDATION
# ========================
class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_VALIDATION_DIR_NAME
        )

        self.validation_report_file_path = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_REPORT_FILE_NAME
        )


# ========================
# DATA TRANSFORMATION
# ========================
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_TRANSFORMATION_DIR_NAME
        )

        self.transformed_train_file_path = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TRAIN_FILE_NAME.replace("csv", "npy")
        )

        self.transformed_test_file_path = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TEST_FILE_NAME.replace("csv", "npy")
        )

        self.transformed_object_file_path = os.path.join(
            self.data_transformation_dir,
            DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            PREPROCSSING_OBJECT_FILE_NAME
        )


# ========================
# MODEL TRAINER
# ========================
class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(
            training_pipeline_config.artifact_dir,
            MODEL_TRAINER_DIR_NAME
        )

        self.trained_model_file_path = os.path.join(
            self.model_trainer_dir,
            MODEL_TRAINER_TRAINED_MODEL_DIR,
            MODEL_FILE_NAME
        )

        self.expected_accuracy = MODEL_TRAINER_EXPECTED_SCORE
        self.model_config_file_path = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH

        # ✅ FIXED ATTRIBUTES (no underscore, no duplicate)
        self.n_estimators = MODEL_TRAINER_N_ESTIMATORS
        self.min_samples_split = MODEL_TRAINER_MIN_SAMPLES_SPLIT
        self.min_samples_leaf = MODEL_TRAINER_MIN_SAMPLES_LEAF
        self.max_depth = MIN_SAMPLES_SPLIT_MAX_DEPTH
        self.criterion = MIN_SAMPLES_SPLIT_CRITERION
        self.random_state = MIN_SAMPLES_SPLIT_RANDOM_STATE


# ========================
# MODEL EVAL
# ========================
@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME


# ========================
# MODEL PUSHER
# ========================
@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME


# ========================
# PREDICTOR
# ========================
@dataclass
class VehiclePredictorConfig:
    model_file_path: str = MODEL_FILE_NAME
    model_bucket_name: str = MODEL_BUCKET_NAME