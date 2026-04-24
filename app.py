"""
Vehicle Insurance Prediction - Flask Backend
Industry-standard API endpoint with JSON support,
error handling, logging, and CORS preparation.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import logging
import traceback
from typing import Dict, Any, Optional

# Configure logging for production insights
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import prediction pipeline components
try:
    from src.pipeline.prediction_pipeline import PredictPipeline, VehicleData
    PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Could not import prediction pipeline: {e}")
    PIPELINE_AVAILABLE = False
    # Fallback mock prediction for demonstration when pipeline is unavailable
    class MockPredictPipeline:
        def predict(self, df):
            # Mock logic based on simple heuristics
            # In real scenario, you would load a trained model
            annual_premium = df['Annual_Premium'].iloc[0] if 'Annual_Premium' in df.columns else 30000
            vehicle_damage = df['Vehicle_Damage'].iloc[0] if 'Vehicle_Damage' in df.columns else 'No'
            previously_insured = df['Previously_Insured'].iloc[0] if 'Previously_Insured' in df.columns else 0
            
            # Simple rule-based mock (for demonstration only)
            if vehicle_damage == 'Yes' and previously_insured == 0:
                return [1]  # Will buy
            elif annual_premium < 20000:
                return [1]
            else:
                return [0]
    
    class MockVehicleData:
        def __init__(self, **kwargs):
            self.data = kwargs
        
        def get_data_as_dataframe(self):
            return pd.DataFrame([self.data])
    
    PredictPipeline = MockPredictPipeline
    VehicleData = MockVehicleData

app = Flask(__name__)

# Optional: Configure for production (remove debug in production)
# app.config['JSON_SORT_KEYS'] = False
# app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True


@app.route("/")
def home():
    """
    Home route - renders the main prediction interface.
    """
    logger.info("Home page accessed")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint - handles both form-urlencoded and JSON requests.
    Returns JSON response for modern frontend or rendered template for traditional POST.
    
    Expected fields:
        - Gender: Male/Female
        - Age: int
        - Driving_License: 0 or 1
        - Region_Code: float/int
        - Previously_Insured: 0 or 1
        - Vehicle_Age: < 1 Year / 1-2 Year / > 2 Years
        - Vehicle_Damage: Yes/No
        - Annual_Premium: float
        - Policy_Sales_Channel: float
        - Vintage: int
    """
    try:
        # Determine request type and extract data
        if request.is_json:
            # Handle JSON request from modern fetch API
            data_payload = request.get_json()
            logger.info(f"JSON prediction request received: {data_payload}")
            
            # Extract fields with fallbacks
            vehicle_data_params = {
                "Gender": data_payload.get("Gender", "Male"),
                "Age": int(data_payload.get("Age", 30)),
                "Driving_License": int(data_payload.get("Driving_License", 1)),
                "Region_Code": float(data_payload.get("Region_Code", 28.0)),
                "Previously_Insured": int(data_payload.get("Previously_Insured", 0)),
                "Vehicle_Age": data_payload.get("Vehicle_Age", "1-2 Year"),
                "Vehicle_Damage": data_payload.get("Vehicle_Damage", "No"),
                "Annual_Premium": float(data_payload.get("Annual_Premium", 25000)),
                "Policy_Sales_Channel": float(data_payload.get("Policy_Sales_Channel", 26.0)),
                "Vintage": int(data_payload.get("Vintage", 120))
            }
        else:
            # Handle traditional form-urlencoded POST (backward compatibility)
            logger.info("Form-encoded prediction request received")
            vehicle_data_params = {
                "Gender": request.form.get("Gender", "Male"),
                "Age": int(request.form.get("Age", 30)),
                "Driving_License": int(request.form.get("Driving_License", 1)),
                "Region_Code": float(request.form.get("Region_Code", 28.0)),
                "Previously_Insured": int(request.form.get("Previously_Insured", 0)),
                "Vehicle_Age": request.form.get("Vehicle_Age", "1-2 Year"),
                "Vehicle_Damage": request.form.get("Vehicle_Damage", "No"),
                "Annual_Premium": float(request.form.get("Annual_Premium", 25000)),
                "Policy_Sales_Channel": float(request.form.get("Policy_Sales_Channel", 26.0)),
                "Vintage": int(request.form.get("Vintage", 120))
            }
        
        # Validate required fields
        required_fields = ['Gender', 'Age', 'Driving_License', 'Region_Code', 
                          'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 
                          'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
        
        for field in required_fields:
            if vehicle_data_params.get(field) is None:
                raise ValueError(f"Missing required field: {field}")
        
        # Create VehicleData object and convert to DataFrame
        try:
            vehicle_data = VehicleData(**vehicle_data_params)
            df = vehicle_data.get_data_as_dataframe()
            logger.info(f"DataFrame created with shape: {df.shape}")
            logger.debug(f"DataFrame columns: {df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error creating VehicleData: {e}")
            # Fallback: Create DataFrame directly if VehicleData constructor fails
            df = pd.DataFrame([vehicle_data_params])
            logger.info("Using fallback DataFrame creation")
        
        # Initialize prediction pipeline and get result
        pipeline = PredictPipeline()
        result = pipeline.predict(df)
        
        # Extract prediction value (handle different return types)
        if isinstance(result, (list, tuple, pd.Series, np.ndarray)):
            prediction_value = int(result[0]) if len(result) > 0 else 0
        elif isinstance(result, (int, float)):
            prediction_value = int(result)
        elif isinstance(result, pd.DataFrame):
            prediction_value = int(result.iloc[0, 0]) if result.shape[0] > 0 else 0
        else:
            prediction_value = int(result) if result is not None else 0
        
        # Prepare human-readable prediction message
        if prediction_value == 1:
            prediction_message = "Customer will BUY insurance ✅"
            risk_level = "High Propensity"
            recommendation = "High likelihood of purchase. Offer competitive rates."
        else:
            prediction_message = "Customer will NOT buy insurance ❌"
            risk_level = "Low Propensity"
            recommendation = "Consider personalized offers or educational content."
        
        logger.info(f"Prediction completed: {prediction_message}")
        
        # Return appropriate response based on request type
        if request.is_json:
            # Return JSON for modern frontend
            return jsonify({
                "success": True,
                "prediction": prediction_message,
                "prediction_code": prediction_value,
                "risk_level": risk_level,
                "recommendation": recommendation,
                "input_summary": {
                    "age": vehicle_data_params["Age"],
                    "vehicle_age": vehicle_data_params["Vehicle_Age"],
                    "annual_premium": vehicle_data_params["Annual_Premium"],
                    "previously_insured": vehicle_data_params["Previously_Insured"]
                }
            }), 200
        else:
            # Return rendered template for traditional form submission
            return render_template("index.html", prediction=prediction_message)
    
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        error_msg = f"Invalid input data: {str(ve)}"
        if request.is_json:
            return jsonify({"success": False, "error": error_msg}), 400
        return render_template("index.html", prediction=error_msg)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        error_msg = f"Prediction failed: {str(e)}"
        
        if request.is_json:
            return jsonify({"success": False, "error": error_msg}), 500
        return render_template("index.html", prediction=error_msg)


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    Returns system status and pipeline availability.
    """
    return jsonify({
        "status": "healthy",
        "pipeline_available": PIPELINE_AVAILABLE,
        "service": "vehicle-insurance-prediction",
        "version": "2.0.0"
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Custom 404 error handler."""
    if request.is_json:
        return jsonify({"success": False, "error": "Endpoint not found"}), 404
    return render_template("index.html", prediction="Error: Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    """Custom 500 error handler."""
    logger.error(f"Internal server error: {error}")
    if request.is_json:
        return jsonify({"success": False, "error": "Internal server error"}), 500
    return render_template("index.html", prediction="Internal server error. Please try again later."), 500


# Optional: Add CORS headers for cross-origin requests if needed
@app.after_request
def add_cors_headers(response):
    """
    Add CORS headers to allow frontend from different origins.
    In production, restrict to specific domains.
    """
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response


if __name__ == "__main__":
    # Run the Flask application
    # For production, use gunicorn or waitress instead of debug=True
    app.run(host="0.0.0.0", port=5000, debug=True)