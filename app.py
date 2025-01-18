from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
# Ensure model.pkl and scaler.pkl are in the same directory as app.py and not ignored in .gitignore
model_path = 'model.pkl'
scaler_path = 'scaler.pkl'

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    model, scaler = None, None
    print(f"Error loading model or scaler: {e}")

@app.route('/')
def home():
    # Serve the HTML form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        # Handle case where the model or scaler failed to load
        return render_template('index.html', 
                               prediction_text='Model or scaler not loaded. Please check the logs.')

    try:
        # Get data from the form
        mass = float(request.form.get('Mass', 0))
        concentration = float(request.form.get('Concentration', 0))
        ph = float(request.form.get('pH', 0))

        # Prepare features for prediction
        features = np.array([[mass, concentration, ph]])
        scaled_features = scaler.transform(features)  # Scale input features using the saved scaler

        # Make prediction
        prediction = model.predict(scaled_features)

        return render_template('index.html',
                               prediction_text=f'Predicted Removal: {prediction[0]:.2f}')
    except ValueError as ve:
        # Handle invalid input
        return render_template('index.html', 
                               prediction_text='Invalid input. Please enter numeric values.')
    except Exception as e:
        # Handle general errors
        return render_template('index.html', 
                               prediction_text=f'Error: {e}')

if __name__ == "__main__":
    # Bind to the PORT environment variable or use default (5000)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
