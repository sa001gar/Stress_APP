import requests

# URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/upload/"

# Path to your CSV file
file_path = "data.csv"

# Send the file in a POST request
with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

# Print the response
print("Response Status Code:", response.status_code)
print("Response JSON:", response.json())
