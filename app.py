from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Initialize the Flask application
app = Flask(__name__)

# Load the trained Random Forest model
# model = joblib.load('random_forest_model.pkl')
model = joblib.load('diabetes_logistic_model.pkl')

# Load the label encoders
label_encoder_diabetes = LabelEncoder()
label_encoder_diabetes.classes_ = np.load('label_encoder_diabetes_classes.npy')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    data = request.form.to_dict()
    feature_names = ['Glucose', 'Age', 'waist', 'Cholesterol', 'BMI', 'Systolic BP', 'Diastolic BP']
    features = pd.DataFrame([[
        float(data['Glucose']),
        int(data['Age']),
        float(data['waist']),
        float(data['Cholesterol']),
        float(data['BMI']),
        float(data['Systolic_BP']),
        float(data['Diastolic_BP'])
    ]], columns=feature_names)

    print(f"DEBUG inputed features:{features}")

    # Make a prediction
    prediction = model.predict(features)
    #
    # 0 - No Diabetes
    # 1 - Diabetes
    probability = model.predict_proba(features)[0][1]  
    prediction_text = label_encoder_diabetes.inverse_transform(prediction)[0]
    print(f"DEBUG prediction: {prediction[0]}")
    probability_text = f'Likelihood of having diabetes: {probability * 100:.2f}%'

    return render_template('index.html', prediction_text=prediction_text, probability_text=probability_text)

if __name__ == '__main__':
    app.run(debug=True)
