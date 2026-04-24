import boto3
import os

from src.constants import (
    AWS_SECRET_ACCESS_KEY_ENV_KEY,
    AWS_ACCESS_KEY_ID_ENV_KEY,
    REGION_NAME
)


class S3Client:
    """
    Singleton S3 client wrapper for AWS connection.
    Ensures only one connection is created for entire project.
    """

    s3_client = None
    s3_resource = None

    def __init__(self, region_name: str = REGION_NAME):

        # Create connection only once (Singleton pattern)
        if S3Client.s3_client is None or S3Client.s3_resource is None:

            # Get credentials from environment variables
            access_key = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
            secret_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)

            # Validate credentials
            if not access_key:
                raise Exception(
                    f"Environment variable {AWS_ACCESS_KEY_ID_ENV_KEY} is not set."
                )

            if not secret_key:
                raise Exception(
                    f"Environment variable {AWS_SECRET_ACCESS_KEY_ENV_KEY} is not set."
                )

            try:
                # Create S3 resource
                S3Client.s3_resource = boto3.resource(
                    "s3",
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region_name
                )

                # Create S3 client
                S3Client.s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region_name
                )

            except Exception as e:
                raise Exception(f"Failed to connect to AWS S3: {str(e)}")

        # Assign to instance
        self.s3_resource = S3Client.s3_resource
        self.s3_client = S3Client.s3_client