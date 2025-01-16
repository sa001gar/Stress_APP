import pandas as pd

def validate_csv_columns(data: pd.DataFrame, required_columns: list) -> bool:
    """Check if the CSV has all required columns."""
    return all(column in data.columns for column in required_columns)

def clean_csv_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the data."""
    # Remove duplicates
    data = data.drop_duplicates()

    # Drop rows with missing critical values
    data = data.dropna(subset=["name", "student_id", "age", "height", "weight", "blood_pressure", "spo2", "sleep_night"])

    # Convert data types if needed
    data["age"] = data["age"].astype(int)
    data["height"] = data["height"].astype(float)
    data["weight"] = data["weight"].astype(float)

    return data
