import requests  # Add this import statement
import numpy as np
from sklearn.metrics import mean_absolute_error
import joblib

# Test the Flask app
def test_flask_app():
    url = "http://127.0.0.1:5000/predict"
    payload = {
        "Mass": 10,
        "Concentration": 5,
        "pH": 7
    }
    response = requests.post(url, data=payload)  # Ensure requests is imported
    assert response.status_code == 200
    assert "Predicted Removal" in response.text

# Test the model directly
def test_model():
    model = joblib.load("model.pkl")  # Adjust path if needed
    X_test = [[10, 5, 7]]  # Example test input
    y_test = [0.8]  # Example expected output
    y_pred = model.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {error}")
    assert error < 0.1  # Ensure predictions are accurate

if __name__ == "__main__":
    test_flask_app()
    test_model()
