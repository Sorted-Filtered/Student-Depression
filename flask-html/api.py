from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load("model.h5") # Change model name

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_data = data["input"]  # Access input data from the request
    prediction = model.predict(input_data)  # Make prediction
    return jsonify({"prediction": str(prediction)})

if __name__ == '__main__':
    app.run(debug=True)