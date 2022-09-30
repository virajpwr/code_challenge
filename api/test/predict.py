import requests
import json


def predict(data):
    url = "http://localhost:8080/predict"
    r = requests.post(url, json=data)
    return r.json()
if __name__ == "__main__":
    # Read test_data.json file
    data = json.load(open("./test_data.json"))
    # Make prediction
    # Save the prediction
    prediction = predict(data)
    print(prediction)
    with open ("../output/prediction.json", "w") as f:
        json.dump(prediction, f)
