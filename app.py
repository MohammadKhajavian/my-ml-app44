import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

app = Flask(__name__)

# Load the model
model = joblib.load('model.pkl')

# Watch for changes to input.csv and retrain the model
class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        global model
        if event.src_path.endswith("input.csv"):
            print("Detected change in input.csv, retraining model...")
            data = pd.read_csv('input.csv')
            X = data[['Mass', 'Concentration', 'pH']]
            y = data['Removal']
            model.fit(X, y)  # Incremental retraining (modify as needed)
            joblib.dump(model, 'model.pkl')
            print("Model retrained and saved.")

# Start a thread for the file watcher
def start_file_watcher():
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=".", recursive=False)
    observer.start()

# Add a route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    X = pd.DataFrame(input_data)
    predictions = model.predict(X).tolist()
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    threading.Thread(target=start_file_watcher, daemon=True).start()

    # Get the port from the environment variable (default to 5000 if not set)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
