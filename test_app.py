import requests
import numpy as np

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
    from train_model import RandomForestRegressor, X_test, y_test
    model = joblib.load("model.pkl")
    y_pred = model.predict(X_test)
    assert np.mean(y_pred - y_test) < 0.1  # Ensuring predictions are accurate

if __name__ == "__main__":
    test_flask_app()
    test_model()
