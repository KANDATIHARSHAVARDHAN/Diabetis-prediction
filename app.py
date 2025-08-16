from flask import Flask, render_template, request
import joblib
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load the encoders, scalers, and model
le = joblib.load("dia_le")
sc = pickle.load(open("dia_scaler.pkl", 'rb'))
model1 = load_model("dia.h5")

app = Flask(__name__)

@app.route('/')
def loadpage():
    return render_template("index.html")

@app.route('/y_predict', methods=["POST", "GET"])
def prediction():
    # Get form data
    gender = request.form["gender"]
    age = float(request.form["age"])
    hypertension = float(request.form["hypertension"])
    heart_disease = float(request.form["heart_disease"])
    smoking_history = float(request.form["smoking_history"])
    bmi = float(request.form["bmi"])
    HbA1c_level = float(request.form["HbA1c_level"])
    blood_glucose_level = float(request.form["blood_glucose_level"])

    # Encode the gender and combine with other features
    gender_encoded = le.transform([gender])[0]
    x_test = [[age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level]]

    # Apply scaling (only to the features used in scaler training)
    x_test_scaled = sc.transform(x_test)

    # Append gender after scaling
    x_test_scaled = np.insert(x_test_scaled, 0, gender_encoded, axis=1)

    # Make prediction
    prediction = model1.predict(x_test_scaled)
    prediction = (prediction > 0.5).astype(int)

    # Convert prediction to text
    if prediction[0][0] == 0:
        text = "negative"
    else:
        text = "positive"

    return render_template("index.html", prediction_text=text)

if __name__ == "__main__":
    app.run(debug=True)
