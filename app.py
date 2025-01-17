from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
# Ensure model.pkl is in the same directory as app.py and not ignored in .gitignore
try:
    model = joblib.load('model.pkl')
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    # Serve the HTML form
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        # Handle case where the model failed to load
        return render_template('index.html', 
                               prediction_text='Model not loaded. Please check the logs.')

    try:
        # Get data from the form
        mass = float(request.form.get('Mass', 0))
        concentration = float(request.form.get('Concentration', 0))
        ph = float(request.form.get('pH', 0))

        # Make prediction
        features = np.array([[mass, concentration, ph]])
        prediction = model.predict(features)

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
