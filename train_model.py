import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load your Excel file
df = pd.read_excel("alumni_data_with_usn_&_distances.xlsx")  # <-- replace with your file name

# Rename columns if needed
# df.columns = ["RSVP", "Past_Attendance", "Distance", "Prediction"]

# Features and label
X = df[["RSVP", "Past_Attendance", "Distance"]]
y = df["Prediction"]

# Scale the Distance
scaler = StandardScaler()
X["Distance"] = scaler.fit_transform(X[["Distance"]])

# Train a simple classifier
model = RandomForestClassifier()
model.fit(X, y)

# Save the model and scaler
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/alumni_attendance_model.pkl")
joblib.dump(scaler, "models/distance_scaler.pkl")

print("Model and scaler saved in the models/ directory.")
