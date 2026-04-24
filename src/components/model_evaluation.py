import sys
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.metrics import f1_score

from src.exception import MyException
from src.logger import logging
from src.constants import TARGET_COLUMN
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact
)

# -----------------------------
# Response DTO
# -----------------------------
@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


# -----------------------------
# Model Evaluation Class
# -----------------------------
class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    # -----------------------------
    # AWS disabled
    # -----------------------------
    def get_best_model(self):
        logging.info("Skipping AWS model fetch")
        return None

    # -----------------------------
    # Preprocessing helpers
    # -----------------------------
    def _map_gender_column(self, df):
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).astype(int)
        return df

    def _create_dummy_columns(self, df):
        return pd.get_dummies(df, drop_first=True)

    def _rename_columns(self, df):
        return df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })

    def _drop_id_column(self, df):
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df

    # -----------------------------
    # Evaluation logic
    # -----------------------------
    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            x = test_df.drop(TARGET_COLUMN, axis=1)
            y = test_df[TARGET_COLUMN]

            # preprocessing
            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            return EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=None,
                is_model_accepted=True,
                difference=trained_model_f1_score
            )

        except Exception as e:
            raise MyException(e, sys)

    # -----------------------------
    # ✅ FIXED: MUST BE INSIDE CLASS
    # -----------------------------
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting Model Evaluation (Skipping AWS)")

            result = self.evaluate_model()

            return ModelEvaluationArtifact(
                is_model_accepted=result.is_model_accepted,
                s3_model_path=None,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy_score=result.difference
            )

        except Exception as e:
            raise MyException(e, sys)