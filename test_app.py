import joblib  # Import joblib
import numpy as np
import pandas as pd
from train_model import RandomForestRegressor, X_test, y_test

# Test the Flask app
def test_flask_app():
    url = "http://127.0.0.1:5000/predict"
    payload = {
        "Mass": 10,
        "Concentration": 5,
        "pH": 7
    }
    response = requests.post(url, data=payload)
    assert response.status_code == 200
    assert "Predicted Removal" in response.text

# Test the model directly
def test_model():
    # Load the trained model
    model = joblib.load("model.pkl")

    # Ensure X_test and y_test are properly defined (from train_model.py)
    # If you cannot import them directly, define them here
    data = pd.read_csv('tests/test_data.csv')  # Make sure the path is correct
    X_test = data[['Mass', 'Concentration', 'pH']]
    y_test = data['Removal']

    # Make predictions
    y_pred = model.predict(X_test)

    # Ensure predictions are accurate
    assert np.mean(y_pred - y_test) < 0.1

if __name__ == "__main__":
    test_flask_app()
    test_model()
