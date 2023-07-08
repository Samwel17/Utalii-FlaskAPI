import numpy as np
import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from os.path import dirname, join, realpath

# Create flask app
app = Flask(__name__)

# Load the model and scaler
model_path = join(dirname(realpath(__file__)), "C:/Users/Samwel Kahungwa/Desktop/FlaskApp/models/lightgbm_model_all.pkl")
scaler_path = join(dirname(realpath(__file__)), "C:/Users/Samwel Kahungwa/Desktop/FlaskApp/models/scaler_all.pkl")
encoder_path = join(dirname(realpath(__file__)), "C:/Users/Samwel Kahungwa/Desktop/FlaskApp/models/one-hot-encoder_all.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
onehotencoder = joblib.load(encoder_path)

@app.route("/")
def home():
    return "Hello, this is the home route!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        # Preprocess the input data
        data = pd.DataFrame(input_data, index=[0])

        continuous_features = data[["Tzmainland", "Zanzibar", "total_number", "int_transport", "accomodation", "food", "local_transport", "sightseeing", "tour_guide", "insurance", "first_trip"]]
        categorical_features = data[['country','age_group','tour_arrangement','travel_with','purpose','main_activity','payment_mode']]
        
        categorical_data = onehotencoder.transform(categorical_features)
        continuous_features = continuous_features.to_numpy()
        preprocessed_data = np.concatenate((continuous_features, categorical_data), axis=1)
        preprocessed_data = scaler.transform(preprocessed_data)

        # Perform prediction
        prediction = model.predict(preprocessed_data)
        output = int(prediction)
        result_dic = {
            1: " From Tsh 0 to Tsh 500,000",
            2: " From Tsh 500,000 to Tsh 1,000,000",
            3: " From Tsh 1,000,000 to Tsh 5,000,000",
            4: " From Tsh 5,000,000 to Tsh 10,000,000",
            5: " From Tsh 10,000,000 and above",
        }
        result = result_dic.get(output, "Unknown")

        return jsonify({'message': result})
    except Exception as e:
        print(str(e))
        return jsonify({'message': 'Error occurred during prediction.'}), 500


if __name__ == '__main__':
    app.run(debug=False)
