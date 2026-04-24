import sys
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import (
    save_object,
    save_numpy_array_data,
    read_yaml_file
)


class DataTransformation:

    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info("Creating preprocessing pipeline")

            num_features = self._schema_config["num_features"]
            mm_columns = self._schema_config["mm_columns"]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("std_scaler", StandardScaler(), num_features),
                    ("minmax_scaler", MinMaxScaler(), mm_columns)
                ],
                remainder="passthrough"
            )

            pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

            logging.info("Preprocessing pipeline created successfully")
            return pipeline

        except Exception as e:
            raise MyException(e, sys) from e

    # ---------------- SAFE TRANSFORMATIONS ----------------

    def _map_gender_column(self, df):
        if "Gender" in df.columns:
            df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
        return df

    def _drop_id_column(self, df):
        drop_col = self._schema_config["drop_columns"]
        if drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
        return df

    def _create_dummy_columns(self, df):
        return pd.get_dummies(df, drop_first=True)

    def _rename_columns(self, df):
        return df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })

    # ---------------- MAIN PIPELINE ----------------

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # FIXED ARTIFACT KEYS
            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # split input/output
            input_train = train_df.drop(columns=[TARGET_COLUMN])
            target_train = train_df[TARGET_COLUMN]

            input_test = test_df.drop(columns=[TARGET_COLUMN])
            target_test = test_df[TARGET_COLUMN]

            # preprocessing
            input_train = self._map_gender_column(input_train)
            input_train = self._drop_id_column(input_train)
            input_train = self._create_dummy_columns(input_train)
            input_train = self._rename_columns(input_train)

            input_test = self._map_gender_column(input_test)
            input_test = self._drop_id_column(input_test)
            input_test = self._create_dummy_columns(input_test)
            input_test = self._rename_columns(input_test)

            # pipeline
            preprocessor = self.get_data_transformer_object()

            train_arr = preprocessor.fit_transform(input_train)
            test_arr = preprocessor.transform(input_test)

            # FORCE NUMERIC TYPE (IMPORTANT FIX)
            train_arr = np.asarray(train_arr, dtype=np.float64)
            test_arr = np.asarray(test_arr, dtype=np.float64)

            # ---------------- SMOTEENN (train only) ----------------
            smt = SMOTEENN(sampling_strategy="minority")

            train_x, train_y = smt.fit_resample(train_arr, target_train)

            test_x = test_arr
            test_y = target_test

            # 🔥 FIX: make everything numpy-safe
            train_y = np.array(train_y).reshape(-1, 1)
            test_y = np.array(test_y).reshape(-1, 1)

            train_arr = np.hstack((train_x, train_y))
            test_arr = np.hstack((test_x, test_y))
            # save artifacts
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )

            logging.info("Data Transformation Completed Successfully")

            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e