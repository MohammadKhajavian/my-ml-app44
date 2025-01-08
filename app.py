from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Initialize Flask app
app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the Random Forest Prediction API! Use the /predict endpoint to make predictions."


@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()
    features = pd.DataFrame(data)

    # Make prediction
    predictions = model.predict(features)

    # Return predictions as JSON
    return jsonify(predictions.tolist())


if __name__ == '__main__':
    app.run(debug=True)
