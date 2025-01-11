import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Load the trained model
model = joblib.load('random_forest_model.pkl')

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json(force=True)
        # Convert input data into a DataFrame
        df = pd.DataFrame(data, index=[0])

        # Make predictions
        prediction = model.predict(df[['Mass', 'Concentration', 'pH']])

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
