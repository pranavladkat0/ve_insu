import os
import sys
import pickle
from io import StringIO
from typing import Union, List

import boto3
import pandas as pd
from pandas import DataFrame, read_csv
from botocore.exceptions import ClientError
from mypy_boto3_s3.service_resource import Bucket

from src.logger import logging
from src.exception import MyException
from src.configuration.aws_connection import S3Client


class SimpleStorageService:
    """
    Handles all AWS S3 operations like upload, download, read, and model handling.
    """

    def __init__(self):
        try:
            s3_client = S3Client()
            self.s3_resource = s3_client.s3_resource
            self.s3_client = s3_client.s3_client
        except Exception as e:
            raise MyException(e, sys)

    # ----------------------------
    # Bucket handling
    # ----------------------------
    def get_bucket(self, bucket_name: str) -> Bucket:
        try:
            return self.s3_resource.Bucket(bucket_name)
        except Exception as e:
            raise MyException(e, sys)

    # ----------------------------
    # Check file existence
    # ----------------------------
    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        try:
            bucket = self.get_bucket(bucket_name)
            files = [obj for obj in bucket.objects.filter(Prefix=s3_key)]
            return len(files) > 0
        except Exception as e:
            raise MyException(e, sys)

    # ----------------------------
    # Read S3 object
    # ----------------------------
    def read_object(self, object_, decode: bool = True, make_readable: bool = False):
        try:
            func = (
                lambda: object_.get()["Body"].read().decode()
                if decode else object_.get()["Body"].read()
            )
            result = func()
            return StringIO(result) if make_readable else result
        except Exception as e:
            raise MyException(e, sys)

    # ----------------------------
    # Get file object
    # ----------------------------
    def get_file_object(self, filename: str, bucket_name: str):
        try:
            bucket = self.get_bucket(bucket_name)
            files = [obj for obj in bucket.objects.filter(Prefix=filename)]

            if len(files) == 1:
                return files[0]
            return files
        except Exception as e:
            raise MyException(e, sys)

    # ----------------------------
    # Load ML model from S3
    # ----------------------------
    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None):
        try:
            model_path = f"{model_dir}/{model_name}" if model_dir else model_name

            file_obj = self.get_file_object(model_path, bucket_name)
            raw_data = self.read_object(file_obj, decode=False)

            model = pickle.loads(raw_data)

            logging.info("Model loaded successfully from S3")
            return model

        except Exception as e:
            raise MyException(e, sys)

    # ----------------------------
    # Upload file to S3
    # ----------------------------
    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True):
        try:
            self.s3_resource.meta.client.upload_file(
                Filename=from_filename,
                Bucket=bucket_name,
                Key=to_filename
            )

            if remove:
                os.remove(from_filename)

        except Exception as e:
            raise MyException(e, sys)

    # ----------------------------
    # Upload DataFrame as CSV
    # ----------------------------
    def upload_df_as_csv(self, df: DataFrame, local_filename: str, bucket_filename: str, bucket_name: str):
        try:
            df.to_csv(local_filename, index=False)
            self.upload_file(local_filename, bucket_filename, bucket_name)

        except Exception as e:
            raise MyException(e, sys)

    # ----------------------------
    # Convert S3 object to DataFrame
    # ----------------------------
    def get_df_from_object(self, object_) -> DataFrame:
        try:
            content = self.read_object(object_, make_readable=True)
            return read_csv(content, na_values="na")
        except Exception as e:
            raise MyException(e, sys)

    # ----------------------------
    # Read CSV from S3
    # ----------------------------
    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        try:
            obj = self.get_file_object(filename, bucket_name)
            return self.get_df_from_object(obj)
        except Exception as e:
            raise MyException(e, sys)