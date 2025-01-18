import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Determine file path dynamically (use relative path)
file_path = 'input.csv'  # Ensure this file is in the same directory as the script

# Load the data
if os.path.exists(file_path):  # Check if file exists
    data = pd.read_csv(file_path)
else:
    raise FileNotFoundError(f"{file_path} not found. Ensure the file is in the correct location.")

# Split the data into features and target
X = data[['Mass', 'Concentration', 'pH']]
y = data['Removal']

# Apply StandardScaler to features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Fit and transform the features

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Save the trained model and scaler to files
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for later use during deployment
