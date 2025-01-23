import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
data = pd.read_csv('input.csv')  # Ensure 'input.csv' is in the repository

# Split features and target
X = data[['Mass', 'Concentration', 'pH']]
y = data['Removal']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the ANN model
model = MLPRegressor(hidden_layer_sizes=(64, 64, 32), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Save the trained model
joblib.dump(model, 'model.pkl')
