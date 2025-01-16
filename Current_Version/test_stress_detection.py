import unittest
import json
from app import app
import joblib
import numpy as np

class TestStressDetection(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.model = joblib.load('stress_model.joblib')
        self.scaler = joblib.load('scaler.joblib')

    def test_predict_endpoint(self):
        test_data = {
            'BVP_mean': 0.5,
            'EDA_phasic_mean': 1.8,
            'EDA_tonic_mean': 1.2,
            'Resp_mean': 0.15,
            'TEMP_mean': 35.8,
            'age': 30,
            'height': 170,
            'weight': 70
        }
        response = self.app.post('/predict', data=json.dumps(test_data), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('stress_level', data)
        self.assertIn(data['stress_level'], [0, 1, 2])

    def test_model_accuracy(self):
        # Load a small subset of your test data
        test_data = np.array([
            [0.5, 1.8, 1.2, 0.15, 35.8, 30, 170, 70],
            [-0.2, 2.1, 0.9, 0.1, 36.2, 25, 165, 60],
            [1.0, 1.5, 1.5, 0.2, 35.5, 40, 180, 80]
        ])
        test_labels = np.array([0, 1, 2])  # Corresponding stress levels

        # Scale the test data
        scaled_test_data = self.scaler.transform(test_data)

        # Make predictions
        predictions = self.model.predict(scaled_test_data)

        # Calculate accuracy
        accuracy = np.mean(predictions == test_labels)
        print(f"Test Accuracy: {accuracy:.2f}")

        # Assert that accuracy is above a certain threshold (e.g., 0.7)
        self.assertGreater(accuracy, 0.7)

if __name__ == '__main__':
    unittest.main()