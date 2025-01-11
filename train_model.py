import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the dataset
file_path = "input.csv"  # Ensure this file is in the same directory
data = pd.read_csv(file_path)

# Separate features and target
X = data[['Mass', 'Concentration', 'pH']]
y = data['Removal']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate evaluation metrics
print("Train Metrics:")
print(f"Mean Squared Error: {mean_squared_error(y_train, y_train_pred)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_train_pred)}")
print(f"R-Squared: {r2_score(y_train, y_train_pred)}")

print("\nTest Metrics:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_test_pred)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_test_pred)}")
print(f"R-Squared: {r2_score(y_test, y_test_pred)}")

# Save the trained model
joblib.dump(model, "random_forest_model.pkl")
print("Model saved as random_forest_model.pkl")
