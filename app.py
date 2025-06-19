from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open(os.path.join('model', 'diabetes_model (3).pkl'), 'rb'))
scaler = pickle.load(open(os.path.join('model', 'scaler (1).pkl'), 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        input_data = scaler.transform([features])
        result = model.predict(input_data)[0]

        prediction = 'High risk of Diabetes 😟' if result == 1 else 'Low risk of Diabetes 😊'
        return render_template('index.html', prediction_text=prediction)

    except Exception as e:
        return render_template('index.html', prediction_text="Error in input: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
