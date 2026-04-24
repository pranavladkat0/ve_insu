import os
import sys
import pandas as pd

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys) from e

    # ---------------- FEATURE STORE ----------------
    def export_data_into_feature_store(self) -> DataFrame:
        try:
            logging.info("Reading data from CSV instead of MongoDB")

            # ✅ Correct path
            data_path = os.path.join("notebook", "data.csv")

            print("Path exists:", os.path.exists(data_path))

            dataframe = pd.read_csv(data_path)

            logging.info(f"Shape of dataframe: {dataframe.shape}")

            feature_store_file_path = self.config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            return dataframe   # ✅ FIXED POSITION

        except Exception as e:
            raise MyException(e, sys) from e   # ✅ FIXED INDENT

    # ---------------- TRAIN TEST SPLIT ----------------
    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        logging.info("Entered split_data_as_train_test method")

        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.config.train_test_split_ratio
            )

            logging.info("Performed train-test split")

            dir_path = os.path.dirname(self.config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test files")

            train_set.to_csv(self.config.training_file_path, index=False, header=True)
            test_set.to_csv(self.config.testing_file_path, index=False, header=True)

            logging.info("Exported train and test files successfully")

        except Exception as e:
            raise MyException(e, sys) from e

    # ---------------- MAIN PIPELINE ----------------
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("Entered initiate_data_ingestion method")

        try:
            dataframe = self.export_data_into_feature_store()

            logging.info("Got data successfully")

            self.split_data_as_train_test(dataframe)

            logging.info("Completed train-test split")

            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.config.training_file_path,
                test_file_path=self.config.testing_file_path,
                feature_store_file_path=self.config.feature_store_file_path
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys) from e