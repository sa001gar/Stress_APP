from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from models.model import predict_stress
from utils.helpers import clean_csv_data, validate_csv_columns
import os

app = FastAPI()

UPLOAD_DIR = "./data/"

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Ensure the uploaded file is a CSV
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    # Save the file locally
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    try:
        # Read the CSV
        data = pd.read_csv(file_path)

        # Validate columns
        required_columns = ["name", "student_id", "age", "height", "weight", "blood_pressure", "spo2", "sleep_night"]
        if not validate_csv_columns(data, required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain the following columns: {', '.join(required_columns)}",
            )

        # Clean the data
        cleaned_data = clean_csv_data(data)

        # Predict stress levels
        predictions = predict_stress(cleaned_data)

        # Format the response
        response = [
            {"name": row["name"], "student_id": row["student_id"], "stress_level": pred}
            for row, pred in zip(cleaned_data.to_dict(orient="records"), predictions)
        ]

        return JSONResponse(content={"data": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up temporary file
        os.remove(file_path)
