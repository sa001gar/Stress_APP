from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the model and scaler
model = joblib.load('stress_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([
        data['BVP_mean'],
        data['EDA_phasic_mean'],
        data['EDA_tonic_mean'],
        data['Resp_mean'],
        data['TEMP_mean'],
        data['age'],
        data['height'],
        data['weight']
    ]).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    return jsonify({'stress_level': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)