import joblib
import pandas as pd

# Load the trained model
MODEL_PATH = "models/stress_model.pkl"
model = joblib.load(MODEL_PATH)

def predict_stress(data):
    """
    Predict stress levels using the trained model.
    Input:
        data: pandas DataFrame with features ["age", "height", "weight", "blood_pressure", "spo2", "sleep_night"]
    Output:
        List of predictions (stress levels: 0, 1, 2)
    """
    features = ["age", "height", "weight", "blood_pressure", "spo2", "sleep_night"]
    predictions = model.predict(data[features])
    return predictions.tolist()
