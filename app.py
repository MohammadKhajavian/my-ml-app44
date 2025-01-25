from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get input data
            data = request.json
            mass = float(data['Mass'])
            concentration = float(data['Concentration'])
            ph = float(data['pH'])

            # Make prediction
            features = [[mass, concentration, ph]]
            prediction = model.predict(features)[0]
            return jsonify({"Prediction": prediction})
        except Exception as e:
            return jsonify({"Error": str(e)}), 400

    return '''
    <h1>Gradient Boosting Prediction System</h1>
    <form method="post" action="/" enctype="application/json">
        <label for="Mass">Mass:</label><br>
        <input type="number" step="0.01" name="Mass"><br>
        <label for="Concentration">Concentration:</label><br>
        <input type="number" step="0.01" name="Concentration"><br>
        <label for="pH">pH:</label><br>
        <input type="number" step="0.1" name="pH"><br><br>
        <input type="submit" value="Predict">
    </form>
    '''

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
