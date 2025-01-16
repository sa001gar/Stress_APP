import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the data
data = pd.read_csv('stress_dataset.csv')

# Select only the specified features
selected_features = ['BVP_mean', 'EDA_phasic_mean', 'EDA_tonic_mean', 'Resp_mean', 'TEMP_mean', 'age', 'height', 'weight']
X = data[selected_features]
y = data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Calculate accuracy
accuracy = model.score(X_test_scaled, y_test)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Save the model and scaler
joblib.dump(model, 'stress_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler saved successfully.")