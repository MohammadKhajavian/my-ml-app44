import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

# For testing (use the relative path to 'test_data.csv' in the 'tests' folder)
#data = pd.read_csv('C:/Users/mokha/Desktop/previous data/Post Doc/Machine learning/My project/CSV file/inputfile/input.csv')

# For deployment (use the relative path to 'input.csv')
data = pd.read_csv('input.csv')  # Uncomment this for deployment

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
