from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input features from the form
        mass = float(request.form['mass'])
        concentration = float(request.form['concentration'])
        ph = float(request.form['ph'])

        # Predict using the model
        features = np.array([[mass, concentration, ph]])
        prediction = model.predict(features)[0]

        return jsonify({
            "Mass": mass,
            "Concentration": concentration,
            "pH": ph,
            "Predicted Removal": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
