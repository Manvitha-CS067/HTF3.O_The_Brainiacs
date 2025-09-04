from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Allows frontend to connect (CORS = Cross-Origin Resource Sharing)

# Load model and scaler
model = joblib.load("alumni_attendance_model.pkl")
scaler = joblib.load("distance_scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    RSVP = data["RSVP"]
    Past_Attendance = data["Past_Attendance"]
    Distance = data["Distance"]

    distance_scaled = scaler.transform([[Distance]])
    input_data = np.array([[RSVP, Past_Attendance, distance_scaled[0][0]]])
    prediction = model.predict(input_data)

    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)


