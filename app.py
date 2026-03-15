import pickle
from flask import Flask, request, jsonify,x_train,y_train
import numpy as np
from sklearn.linear_model import LogisticRegression
app = Flask(__name__)
# Initialize the Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train, y_train)

print("Logistic Regression model initialized and trained successfully.")

# Initialize the Flask application

# Load the trained model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
print("Trained Logistic Regression model loaded.")

# Load the fitted scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
print("Fitted StandardScaler loaded.")

# You can add Flask routes and other application logic here later
# For now, we just set up the app and load the model/scaler.
if __name__ == "__main__":
    app.run(debug=True)
