from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the combined model and scaler
try:
    combined = joblib.load('model_and_scaler.pkl')
    model = combined['model']
    scaler = combined['scaler']
except Exception as e:
    model = None
    scaler = None
    print(f"Error loading model or scaler: {e}")

@app.route('/')
def index():
    return render_template('index.html')  # Ensure you have a corresponding `index.html` file.

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return "Model or scaler not loaded. Please check the logs."

    try:
        # Get input features from the form
        features = [float(x) for x in request.form.values()]
        features_array = np.array(features).reshape(1, -1)

        # Scale the input features
        scaled_features = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(scaled_features)
        return jsonify({'Prediction': float(prediction[0])})
    except Exception as e:
        return jsonify({'Error': str(e)})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get the PORT from environment variables or default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)

