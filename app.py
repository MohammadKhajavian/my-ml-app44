from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return '''
    <h1>ML Model Prediction</h1>
    <form action="/predict" method="post">
      Mass: <input type="number" step="0.01" name="mass"><br>
      Concentration: <input type="number" name="concentration"><br>
      pH: <input type="number" step="0.1" name="ph"><br>
      <button type="submit">Predict</button>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data
    mass = float(request.form['mass'])
    concentration = float(request.form['concentration'])
    ph = float(request.form['ph'])

    # Make prediction
    prediction = model.predict([[mass, concentration, ph]])[0]
    return jsonify({'Removal Prediction': round(prediction, 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
