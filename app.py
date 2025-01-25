from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extract features
        features = np.array([[data['Mass'], data['Concentration'], data['pH']]])
        # Scale features
        features_scaled = scaler.transform(features)
        # Make prediction
        prediction = model.predict(features_scaled)
        return jsonify({'Removal': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
