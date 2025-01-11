from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "Random Forest Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    mass = data['Mass']
    concentration = data['Concentration']
    ph = data['pH']
    # Make prediction
    features = np.array([[mass, concentration, ph]])
    prediction = model.predict(features)
    return jsonify({'predicted_removal': prediction[0]})

if __name__ == "__main__":
    # Get the PORT environment variable (default to 5000 if not set)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
