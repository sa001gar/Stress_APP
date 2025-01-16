from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import os

# Import custom modules with error handling
try:
    from models.model import predict_stress
    from utils.helpers import clean_csv_data, validate_csv_columns
except ImportError as e:
    raise ImportError(f"Required modules not found: {str(e)}. Please ensure 'models' and 'utils' packages are properly installed.")

app = FastAPI(title="Stress Detection API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
UPLOAD_DIR = "./data/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/", response_model=dict)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a CSV file for stress prediction.
    
    Args:
        file (UploadFile): CSV file with required columns
        
    Returns:
        JSONResponse: Predicted stress levels for each student
    """
    # Validate file extension
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    # Create unique filename to avoid conflicts
    file_path = os.path.join(UPLOAD_DIR, f"temp_{os.urandom(8).hex()}.csv")
    
    try:
        # Save the file locally
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
            
        with open(file_path, "wb") as f:
            f.write(contents)

        # Read the CSV with error handling
        try:
            data = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            raise HTTPException(status_code=400, detail="The CSV file is empty.")
        except pd.errors.ParserError:
            raise HTTPException(status_code=400, detail="Invalid CSV format.")

        # Validate columns
        required_columns = [
            "name", "student_id", "age", "height", 
            "weight", "blood_pressure", "spo2", "sleep_night"
        ]
        
        if not validate_csv_columns(data, required_columns):
            raise HTTPException(
                status_code=400,
                detail=f"CSV must contain all required columns: {', '.join(required_columns)}"
            )

        # Validate data is not empty
        if data.empty:
            raise HTTPException(status_code=400, detail="The CSV file contains no data.")

        # Clean the data
        try:
            cleaned_data = clean_csv_data(data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error cleaning data: {str(e)}")

        # Predict stress levels
        try:
            predictions = predict_stress(cleaned_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error predicting stress levels: {str(e)}")

        # Format the response
        try:
            response = [
                {
                    "name": str(row["name"]),  # Convert to string to handle non-string names
                    "student_id": str(row["student_id"]),  # Convert to string to handle numeric IDs
                    "stress_level": float(pred)  # Ensure prediction is float
                }
                for row, pred in zip(cleaned_data.to_dict(orient="records"), predictions)
            ]
            
            return JSONResponse(content={"data": response})
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error formatting response: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
        
    finally:
        # Clean up temporary file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {file_path}: {str(e)}")

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running"""
    return {"status": "healthy"}