import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load data
data = pd.read_csv('input.csv')

# Split features and target
X = data[['Mass', 'Concentration', 'pH']]
y = data['Removal']

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')
