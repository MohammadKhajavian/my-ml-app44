from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')

@app.route('/', methods=['GET'])
def home():
    return '''
    <h1>Water Treatment Prediction</h1>
    <form action="/predict" method="post">
        Mass: <input type="text" name="Mass"><br>
        Concentration: <input type="text" name="Concentration"><br>
        pH: <input type="text" name="pH"><br>
        <input type="submit" value="Predict">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        mass = float(request.form['Mass'])
        concentration = float(request.form['Concentration'])
        ph = float(request.form['pH'])

        # Prepare input data
        input_data = np.array([[mass, concentration, ph]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return jsonify({'Prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
