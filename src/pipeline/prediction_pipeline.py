import sys
import pandas as pd

from src.exception import MyException
from src.utils.main_utils import load_object


class PredictPipeline:

    def __init__(self):
        try:
            self.model_path = "saved_models/model.pkl"

            # 🎯 threshold for classification
            self.threshold = 0.30  # adjust 0.25–0.4 for your dataset

        except Exception as e:
            raise MyException(e, sys)

    # -----------------------------
    # prediction
    # -----------------------------
    def predict(self, input_df: pd.DataFrame):
        try:
            model_obj = load_object(self.model_path)

            preprocessing = model_obj.preprocessing_object
            model = model_obj.trained_model_object

            # transform input
            processed_data = preprocessing.transform(input_df)

            # probability
            probability = model.predict_proba(processed_data)[:, 1]

            # final decision using threshold
            prediction = [
                1 if p >= self.threshold else 0
                for p in probability
            ]

            # 🔥 convert to user-friendly labels
            result = [
                "Likely to purchase" if p == 1 else "Unlikely to purchase"
                for p in prediction
            ]

            return result, probability

        except Exception as e:
            raise MyException(e, sys)