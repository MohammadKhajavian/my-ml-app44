from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.joblib')

# Endpoint for the main prediction page
@app.route('/')
def home():
    return '''
    <h1>Gradient Boosting Prediction</h1>
    <form action="/predict" method="post">
      Mass: <input type="number" step="0.01" name="mass"><br>
      Concentration: <input type="number" step="0.1" name="concentration"><br>
      pH: <input type="number" step="0.1" name="ph"><br>
      <button type="submit">Predict</button>
    </form>
    '''

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input values
    mass = float(request.form['mass'])
    concentration = float(request.form['concentration'])
    ph = float(request.form['ph'])

    # Make prediction
    prediction = model.predict([[mass, concentration, ph]])[0]
    return jsonify({'Predicted Removal': round(prediction, 2)})

# Endpoint for real-time data update (from HYSYS or other sources)
@app.route('/update', methods=['POST'])
def update_model():
    # Assuming JSON data with keys 'Mass', 'Concentration', 'pH', and 'Removal'
    data = request.get_json()
    new_data = pd.DataFrame(data)

    # Split features and target
    X = new_data[['Mass', 'Concentration', 'pH']]
    y = new_data['Removal']

    # Incremental training (simulated here by re-training)
    global model
    model.fit(X, y)
    joblib.dump(model, 'model.joblib')
    return "Model updated with new data", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
