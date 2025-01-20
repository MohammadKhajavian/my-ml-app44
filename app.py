from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the app
app = Flask(__name__)

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        mass = float(request.form['Mass'])
        concentration = float(request.form['Concentration'])
        pH = float(request.form['pH'])
        
        # Prepare the input array
        features = np.array([[mass, concentration, pH]])
        
        # Scale the input
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        return render_template('index.html', prediction_text=f'Predicted Removal: {prediction[0]:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
