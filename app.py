from flask import Flask, request, jsonify
import joblib
from tensorflow.keras.models import load_model
from scipy.optimize import minimize

app = Flask(__name__)

# Load model and scaler
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

# Define the optimization function
def objective(features):
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features, verbose=0)
    return abs(65 - prediction[0][0])

# Bounds for the features
bounds = [(0.01, 0.18), (0, 300), (2, 9)]

@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json
    def optimize_features():
        result = minimize(objective, [data['Mass'], data['Concentration'], data['pH']], bounds=bounds)
        return result.x

    optimized = optimize_features()
    return jsonify({
        'optimized_mass': optimized[0],
        'optimized_concentration': optimized[1],
        'optimized_pH': optimized[2]
    })

if __name__ == '__main__':
    app.run(debug=True)
