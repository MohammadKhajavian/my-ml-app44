import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# For testing (use the relative path to 'test_data.csv' in the 'tests' folder)
data = pd.read_csv('tests/test_data.csv')  # Adjust this path as needed for testing

# For deployment (use the relative path to 'input.csv')
# data = pd.read_csv('input.csv')  # Uncomment this for deployment

# Split the data into features and target
X = data[['Mass', 'Concentration', 'pH']]
y = data['Removal']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Save the trained model to a file
joblib.dump(model, 'model.pkl')
