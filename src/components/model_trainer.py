import sys
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from imblearn.over_sampling import SMOTE   # ⭐ ADDED

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)
from src.entity.estimator import MyModel


class ModelTrainer:

    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    # -----------------------------
    # Model training
    # -----------------------------
    def get_model_object_and_report(self, train: np.array, test: np.array):

        try:
            logging.info("Training RandomForest model (SMOTE + Balanced)")

            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            # 🔥 STEP 1: APPLY SMOTE (ONLY TRAIN DATA)
            smote = SMOTE(random_state=42)
            x_train, y_train = smote.fit_resample(x_train, y_train)

            logging.info(f"After SMOTE shape: {x_train.shape}, {y_train.shape}")

            # 🔥 STEP 2: MODEL
            model = RandomForestClassifier(
                n_estimators=self.model_trainer_config.n_estimators,
                min_samples_split=self.model_trainer_config.min_samples_split,
                min_samples_leaf=self.model_trainer_config.min_samples_leaf,
                max_depth=self.model_trainer_config.max_depth,
                criterion=self.model_trainer_config.criterion,
                random_state=self.model_trainer_config.random_state,
                class_weight="balanced"
            )

            model.fit(x_train, y_train)

            # predictions
            y_pred = model.predict(x_test)

            # metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)

            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"F1 Score: {f1}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")

            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall
            )

            return model, metric_artifact

        except Exception as e:
            raise MyException(e, sys)

    # -----------------------------
    # Training pipeline
    # -----------------------------
    def initiate_model_trainer(self):

        try:
            logging.info("Starting Model Trainer")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )

            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            model, metric_artifact = self.get_model_object_and_report(
                train_arr,
                test_arr
            )

            preprocessing_obj = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            # training accuracy check
            train_acc = accuracy_score(
                train_arr[:, -1],
                model.predict(train_arr[:, :-1])
            )

            logging.info(f"Training Accuracy: {train_acc}")

            if train_acc < self.model_trainer_config.expected_accuracy:
                raise Exception(f"Model accuracy too low: {train_acc}")

            my_model = MyModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=model
            )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                my_model
            )

            logging.info("Model saved successfully")

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

        except Exception as e:
            raise MyException(e, sys)