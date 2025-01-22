import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib
from scipy.optimize import minimize

# Custom objects for loading the model
custom_objects = {'mse': MeanSquaredError()}

# Load model and scaler
model = load_model('model.h5', custom_objects=custom_objects)
scaler = joblib.load('scaler.pkl')

# Define the target range
target_min, target_max = 60, 70

# Define the optimization function
def objective(features):
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)
    return abs((target_min + target_max) / 2 - prediction[0][0])

# Bounds for the features
bounds = [(0.01, 0.18), (0, 300), (2, 9)]

def optimize_features():
    result = minimize(objective, [0.1, 150, 5], bounds=bounds)
    return result.x

if __name__ == '__main__':
    # Example usage
    optimized_features = optimize_features()
    print("Optimized Features:", optimized_features)
