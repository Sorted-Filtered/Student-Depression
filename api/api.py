from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# Load the model
model = joblib.load("model.h5")

@app.route("/")
def welcome():
    """List all available api routes."""
    return (
        f"Available Routes:<br/>"
        f"/predict<br/>"
    )

@app.route("/predict", methods=["POST"])
def predict():
    # Isolate needed data from json
    data = request.get_json()
    input_data = data["input"]
    print(input_data)

    # Load scaler from file
    scaler = joblib.load("scaler.bin")

    # Create list and append non-dummified values as floats
    input_list = []
    input_list.append(input_data[0:6])

    # Append values to input_list based on profession dummy category
    if input_data[7] == "Civil Engineer":
        input_list.append([True, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[7] == "Content Writer":
        input_list.append([False, True, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[7] == "Digital Marketer":
        input_list.append([False, False, True, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[7] == "Education Consultant":
        input_list.append([False, False, False, True, False, False, False, False, False, False, False, False, False, False])
    elif input_data[7] == "UX/UI Designer":
        input_list.append([False, False, False, False, True, False, False, False, False, False, False, False, False, False])
    elif input_data[7] == "Architect":
        input_list.append([False, False, False, False, False, True, False, False, False, False, False, False, False, False])
    elif input_data[7] == "Chef":
        input_list.append([False, False, False, False, False, False, True, False, False, False, False, False, False, False])
    elif input_data[7] == "Doctor":
        input_list.append([False, False, False, False, False, False, False, True, False, False, False, False, False, False])
    elif input_data[7] == "Entrepreneur":
        input_list.append([False, False, False, False, False, False, False, False, True, False, False, False, False, False])
    elif input_data[7] == "Lawyer":
        input_list.append([False, False, False, False, False, False, False, False, False, True, False, False, False, False])
    elif input_data[7] == "Lawyer":
        input_list.append([False, False, False, False, False, False, False, False, False, False, True, False, False, False])
    elif input_data[7] == "Lawyer":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, True, False, False])
    elif input_data[7] == "Lawyer":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, True, False])
    else:
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, True])

    # Append values to input_list based on city dummy category
    if input_data[8] == "Less Dehli":
        input_list.append([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Less than 5 Kalyan":
        input_list.append([False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "3.0'":
        input_list.append([False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Agra":
        input_list.append([False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Ahmedabad":
        input_list.append([False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Bangalore":
        input_list.append([False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Bhavna":
        input_list.append([False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Bhopal":
        input_list.append([False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Chennai":
        input_list.append([False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "City":
        input_list.append([False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Delhi":
        input_list.append([False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Faridabad":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Gaurav":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Ghaziabad":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Harsh":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Harsha":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Hyderabad":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Indore":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Jaipur":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Kalyan":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Kanpur":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Khaziabad":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Kibara":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Kolkata":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Lucknow":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Ludhiana":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "M.Com":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "M.Tech":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "ME":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Meerut":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Mihir":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Mira":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Mumbai":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Nagpur":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Nalini":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Nalyan":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Nandini":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Nashik":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Patna":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Pune":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Rajkot":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Rashi":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Reyansh":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Saanvi":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Srinagar":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False])
    elif input_data[8] == "Surat":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False])
    elif input_data[8] == "Thane":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False])
    elif input_data[8] == "Vaanya":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False])
    elif input_data[8] == "Vadodara":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False])
    elif input_data[8] == "Varanasi":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False])
    elif input_data[8] == "Vasai-Virar":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False])
    else:
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False])
    
    # Append values to input_list based on Gender dummy category
    if input_data[9] == "Female":
        input_list.append([True, False])
    else:
        input_list.append([False, True])

    # Append values to input_list based on sleep duration dummy category
    if input_data[10] == "5-6 hours":
        input_list.append([True, False, False, False, False])
    elif input_data[10] == "7-8 hours":
        input_list.append([False, True, False, False, False])
    elif input_data[10] == "Less than 5 hours":
        input_list.append([False, False, True, False, False])
    elif input_data[10] == "More than 8 hours":
        input_list.append([False, False, False, True, False])
    else:
        input_list.append([False, False, False, False, True])

    # Append values to input_list based on dietary habits dummy category
    if input_data[11] == "Healthy":
        input_list.append([True, False, False, False])
    elif input_data[11] == "Moderate":
        input_list.append([False, True, False, False])
    elif input_data[11] == "Others":
        input_list.append([False, False, True, False])
    else:
        input_list.append([False, False, False, True])

    # Append values to input_list based on profession dummy category
    if input_data[12] == "Class 12":
        input_list.append([True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "B.Arch":
        input_list.append([False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "B.Com":
        input_list.append([False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "B.Ed":
        input_list.append([False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "B.Pharm":
        input_list.append([False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "B.Tech":
        input_list.append([False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "BA":
        input_list.append([False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "BBA":
        input_list.append([False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "BCA":
        input_list.append([False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "BE":
        input_list.append([False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "BHM":
        input_list.append([False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "BSc":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "LLB":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "LLM":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "M.Com":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "M.Ed":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "M.Pharm":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "M.Tech":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "MA":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False, False])
    elif input_data[12] == "MBA":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False, False])
    elif input_data[12] == "MBBS":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False, False])
    elif input_data[12] == "MCA":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False, False])
    elif input_data[12] == "MD":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False, False])
    elif input_data[12] == "ME":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False, False])
    elif input_data[12] == "MHM":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False, False])
    elif input_data[12] == "MSc":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False, False])
    elif input_data[12] == "Others":
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, False])
    else:
        input_list.append([False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True])

    # Append values to input_list based on suicidal thoughts dummy category
    if input_data[13] == "No":
        input_list.append([True, False])
    else:
        input_list.append([False, True])

    # Append values to input_list based on family mental illness dummy category
    if input_data[14] == "No":
        input_list.append([True, False])
    else:
        input_list.append([False, True])

    # Append values to input_list based on financial stress dummy category
    if input_data[15] == "1":
        input_list.append([True, False, False, False, False, False])
    elif input_data[15] == "2":
        input_list.append([False, True, False, False, False, False])
    elif input_data[15] == "3":
        input_list.append([False, False, True, False, False, False])
    elif input_data[15] == "4":
        input_list.append([False, False, False, True, False, False])
    elif input_data[15] == "5":
        input_list.append([False, False, False, False, True, False])
    else:
        input_list.append([False, False, False, False, False, True])

    print(input_list)

    # Clean function to remove nested lists from input_list
    def flatten_list(nested_list):
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(flatten_list(item))
            else:
                flat_list.append(float(item))
        return flat_list

    clean_input_list = flatten_list(input_list)
    clean_input_array = np.array(clean_input_list)
    clean_input_array.shape = (1, 122)
    print(clean_input_array)

    #Scale list using saved scaler from model creation
    scaled_input_array = scaler.transform(clean_input_array)
    print(scaled_input_array)

    # Make prediction
    prediction = model.predict(scaled_input_array)
    if prediction[0] == 0:
        output = "Do you likely have depression?: No"
    elif prediction[0] == 1:
        output = "Do you likely have depression?: Yes"
    print(output)
    return jsonify({"prediction": output})

if __name__ == '__main__':
    app.run(debug=True)