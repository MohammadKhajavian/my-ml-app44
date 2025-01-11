from flask import Flask, request, jsonify
import joblib
import numpy as np

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
    app.run(debug=True)
