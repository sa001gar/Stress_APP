from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from models.model import predict_stress
from utils.helpers import clean_csv_data, validate_csv_columns
import os
import logging

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Upload directory
UPLOAD_DIR = "./data/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Ensure the uploaded file is a CSV
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    # Save the file locally
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
        logging.info(f"File saved to {file_path}")

        # Read the CSV
        data = pd.read_csv(file_path)

        # Validate required columns
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

    except pd.errors.ParserError as e:
        logging.error(f"Error reading CSV file: {str(e)}")
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid CSV.")
    except HTTPException as e:
        logging.error(f"Validation error: {str(e.detail)}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Temporary file {file_path} deleted")
