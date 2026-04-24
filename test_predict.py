from src.pipeline.prediction_pipeline import PredictPipeline, VehicleData
# create sample input
data = VehicleData(
    Gender="Male",
    Age=30,
    Driving_License=1,
    Region_Code=28,
    Previously_Insured=0,
    Vehicle_Age="1-2 Year",
    Vehicle_Damage="Yes",
    Annual_Premium=30000,
    Policy_Sales_Channel=26,
    Vintage=200
)

df = data.get_data_as_dataframe()

pipeline = PredictPipeline()
result = pipeline.predict(df)

print("Prediction:", result)