import sys
import pandas as pd

from src.exception import MyException
from src.utils.main_utils import load_object


class PredictPipeline:

    def __init__(self):
        try:
            self.model_path = "saved_models/model.pkl"

            # ✅ include 'id'
            self.expected_columns = [
                'id',
                'Gender', 'Age', 'Driving_License', 'Region_Code',
                'Previously_Insured', 'Annual_Premium',
                'Policy_Sales_Channel', 'Vintage',
                'Vehicle_Age_lt_1_Year',
                'Vehicle_Age_gt_2_Years',
                'Vehicle_Damage_Yes'
            ]

        except Exception as e:
            raise MyException(e, sys)

    # -----------------------------
    # SAME preprocessing as training
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

    def _drop_mongo_id(self, df):
        # ✅ only drop MongoDB _id, NOT 'id'
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df

    # -----------------------------
    # ALIGN COLUMNS
    # -----------------------------
    def _align_columns(self, df):
        for col in self.expected_columns:
            if col not in df.columns:
                df[col] = 0

        # Ensure correct order
        df = df[self.expected_columns]

        return df

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self, input_df: pd.DataFrame):
        try:
            model = load_object(self.model_path)

            # preprocessing
            input_df = self._map_gender_column(input_df)
            input_df = self._drop_mongo_id(input_df)
            input_df = self._create_dummy_columns(input_df)
            input_df = self._rename_columns(input_df)

            # 🔥 important
            input_df = self._align_columns(input_df)

            prediction = model.predict(input_df)

            return prediction

        except Exception as e:
            raise MyException(e, sys)


# -----------------------------
# Input Data Class
# -----------------------------
class VehicleData:

    def __init__(
        self,
        Gender,
        Age,
        Driving_License,
        Region_Code,
        Previously_Insured,
        Vehicle_Age,
        Vehicle_Damage,
        Annual_Premium,
        Policy_Sales_Channel,
        Vintage
    ):
        self.Gender = Gender
        self.Age = Age
        self.Driving_License = Driving_License
        self.Region_Code = Region_Code
        self.Previously_Insured = Previously_Insured
        self.Vehicle_Age = Vehicle_Age
        self.Vehicle_Damage = Vehicle_Damage
        self.Annual_Premium = Annual_Premium
        self.Policy_Sales_Channel = Policy_Sales_Channel
        self.Vintage = Vintage

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "id": [1],  # ✅ REQUIRED (dummy value)
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Driving_License": [self.Driving_License],
                "Region_Code": [self.Region_Code],
                "Previously_Insured": [self.Previously_Insured],
                "Vehicle_Age": [self.Vehicle_Age],
                "Vehicle_Damage": [self.Vehicle_Damage],
                "Annual_Premium": [self.Annual_Premium],
                "Policy_Sales_Channel": [self.Policy_Sales_Channel],
                "Vintage": [self.Vintage],
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise MyException(e, sys)