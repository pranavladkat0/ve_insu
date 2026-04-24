import sys
import shutil
import os

from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig


class ModelPusher:

    def __init__(
        self,
        model_evaluation_artifact: ModelEvaluationArtifact,
        model_pusher_config: ModelPusherConfig
    ):
        try:
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
        except Exception as e:
            raise MyException(e, sys) from e


    # -----------------------------
    # Save model locally (instead of S3)
    # -----------------------------
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Starting Model Pusher (LOCAL MODE)")

            # check if model is accepted
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model rejected, not saving")
                return None

            source_path = self.model_evaluation_artifact.trained_model_path

            # ✅ FIX: Ensure this is treated as a DIRECTORY
            target_dir = self.model_pusher_config.s3_model_key_path

            # if user mistakenly passes "model.pkl", fix it
            if target_dir.endswith(".pkl"):
                target_dir = os.path.dirname(target_dir)

            # fallback if empty
            if target_dir == "":
                target_dir = "saved_models"

            # final model path
            target_path = os.path.join(target_dir, "model.pkl")

            # create directory
            os.makedirs(target_dir, exist_ok=True)

            # copy model
            shutil.copy(source_path, target_path)

            logging.info(f"Model saved locally at: {target_path}")

            artifact = ModelPusherArtifact(
                bucket_name="local_storage",
                s3_model_path=target_path
            )

            return artifact

        except Exception as e:
            raise MyException(e, sys) from e