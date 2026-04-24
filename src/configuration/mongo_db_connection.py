import os
import sys
import pymongo
import certifi
from dotenv import load_dotenv   # ✅ STEP 3 ADDED

from src.exception import MyException
from src.logger import logging
from src.constants import DATABASE_NAME, MONGODB_URL_KEY

# ✅ Load .env file BEFORE reading environment variables
load_dotenv()

# Load certificate authority for secure MongoDB connection
ca = certifi.where()


class MongoDBClient:
    """
    MongoDB client for connecting to database
    """

    client = None

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        try:
            # Create connection only once (singleton pattern)
            if MongoDBClient.client is None:

                # Get MongoDB URL from environment
                mongo_db_url = os.getenv(MONGODB_URL_KEY)

                if mongo_db_url is None:
                    raise Exception(
                        f"Environment variable '{MONGODB_URL_KEY}' is not set. "
                        f"Please check your .env file."
                    )

                # Connect to MongoDB
                MongoDBClient.client = pymongo.MongoClient(
                    mongo_db_url,
                    tlsCAFile=ca
                )

            # Assign database
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name

            logging.info("MongoDB connection successful")

        except Exception as e:
            raise MyException(e, sys) from e