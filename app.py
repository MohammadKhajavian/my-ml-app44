from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')


@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML form


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        mass = float(request.form['Mass'])
        concentration = float(request.form['Concentration'])
        ph = float(request.form['pH'])

        # Make prediction
        features = np.array([[mass, concentration, ph]])
        prediction = model.predict(features)

        return render_template('index.html',
                               prediction_text=f'Predicted Removal: {prediction[0]:.2f}')
    except Exception as e:
        return render_template('index.html',
                               prediction_text='Error: ' + str(e))


if __name__ == "__main__":
    # Get the PORT environment variable (default to 5000 if not set)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
