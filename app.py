from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained Random Forest model
# model = joblib.load('random_forest_model.pkl')
model = joblib.load('diabetes_logistic_modelMINE.pkl')
# Standard scaler used during model training
scaler = joblib.load('scaler.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    data = request.form.to_dict()
    features = np.array([[
        float(data['Cholesterol']),
        float(data['Glucose']),
        float(data['HDL_Chol']),
        float(data['Chol_HDL_ratio']),
        int(data['Age']),
        int(data['Gender']),
        float(data['Height']),
        float(data['Weight']),
        float(data['BMI']),
        float(data['Systolic_BP']),
        float(data['Diastolic_BP']),
        float(data['waist']),
        float(data['hip']),
        float(data['Waist_hip_ratio']),
        int(data['BMI_Category']),
        int(data['BP_Category']),
        float(data['Waist_to_Height'])
    ]])

    # Scale the features
    features_scaled = scaler.transform(features)

    # Make a prediction
    prediction = model.predict(features_scaled)
    prediction_text = 'Diabetes' if prediction[0] == 0 else 'No Diabetes'

    return render_template('index.html', prediction_text=f'Prediction: {prediction_text}')

if __name__ == '__main__':
    app.run(debug=True)
