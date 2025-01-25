import os
import pandas as pd
import threading
from flask import Flask, request, jsonify, render_template
import joblib
from tensorflow.keras.models import load_model
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

app = Flask(__name__)

# Load the ANN model and scaler
model = load_model('ann_model.keras')
scaler = joblib.load('scaler.pkl')

# Monitor input.csv for real-time updates
class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith("input.csv"):
            print("Detected change in input.csv, updating the model...")
            data = pd.read_csv('input.csv')
            X = data[['Mass', 'Concentration', 'pH']]
            y = data['Removal']
            X_scaled = scaler.transform(X)
            model.fit(X_scaled, y, epochs=10, batch_size=8, verbose=0)
            model.save('ann_model.h5')
            print("Model updated successfully.")

def start_file_watcher():
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=False)
    observer.start()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        input_data = request.get_json()
        X_input = pd.DataFrame(input_data)
        X_scaled = scaler.transform(X_input)
        predictions = model.predict(X_scaled).flatten().tolist()
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    threading.Thread(target=start_file_watcher, daemon=True).start()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
