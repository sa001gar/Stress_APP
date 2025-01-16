import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Generate synthetic data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)

    data = {
        "age": np.random.randint(18, 60, n_samples),
        "height": np.random.randint(150, 200, n_samples),
        "weight": np.random.randint(50, 100, n_samples),
        "blood_pressure": np.random.randint(100, 140, n_samples),
        "spo2": np.random.randint(90, 100, n_samples),
        "sleep_night": np.random.randint(4, 9, n_samples),
    }
    # Randomly assign stress levels (0, 1, 2)
    data["stress_level"] = np.random.choice([0, 1, 2], n_samples)

    return pd.DataFrame(data)

# Train and save the model
def train_and_save_model():
    # Generate data
    df = generate_synthetic_data()
    
    # Features and labels
    X = df[["age", "height", "weight", "blood_pressure", "spo2", "sleep_night"]]
    y = df["stress_level"]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Save the trained model
    joblib.dump(model, "stress_model.pkl")
    print("Model saved as models/stress_model.pkl")

if __name__ == "__main__":
    train_and_save_model()
