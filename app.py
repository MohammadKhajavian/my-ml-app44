from flask import Flask, request, jsonify, render_template
import joblib
from tensorflow.keras.models import load_model
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Load model and scaler
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = np.array([[data['Mass'], data['Concentration'], data['pH']]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return jsonify({'Removal': float(prediction[0][0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)