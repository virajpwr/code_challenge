# from model import SentimentModel, SentimentQueryModel
from fastapi import FastAPI
import uvicorn
from flask import Flask, jsonify, make_response, request
from imports import *

app = Flask(__name__)


@app.route('/score', methods=['POST'])
def score():
    with open(r".\config.yml", 'r') as f:
        config = yaml.safe_load(f)
    logger = logs(path="logs/", file="model_training.logs")
    logger.info(
        "Reading the final data from the data/processed folder and training the model")
    content = request.json
    print(content)
    df = pd.DataFrame(content, index=[0])
    # df = pd.DataFrame.from_dict(content, orient='index').T
    # print(type(df))
    df_processed = pd.read_csv(r'./data/preprocessed_data.csv')
    df_test = df_processed.append(df, ignore_index=True)
    model = joblib.load(r'./models/rf.dat')
    # df = pd.DataFrame(content, index=[0])  # convert to dataframe
    feat_engg = FeatEngg(df_test, config, logger)
    df_test = feat_engg.split_datetime_col()  # split datetime column
    df_test = feat_engg.cal_time_diff()
    # label encode the categorical columns
    df_test = feat_engg.categorify_columns()
    # target encode the categorical columns
    df_test = feat_engg.target_encode_columns()
    # count encode the categorical columns
    df_test = feat_engg.count_encode_columns()
    # transform to gaussian distribution.
    df = feat_engg.transforming_target_continuous()
    logger.info(
        "df transformation done")
    df_test = df_test[config['required_cols_prediction']]
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict(df_test)
    return {'target': prediction[0]}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)


# Create a flask app for deployment using flask_restful

# app = Flask(__name__)
# api = Api(app)

# # Load the model from the file
# # model = pickle.load(open("model.pkl", "rb"))
# model = load("./models/rf.dat")

# Create a route for the prediction

#########

# app = FastAPI()
# model = joblib.load(r'./models/rf.dat')


# @app.post('/predict')
# def predict(data):
#     config = load_config(r".\config.yml")
#     df = data.to_json(orient='records')
#     feat_engg = FeatEngg(df, config, logger)
#     df = feat_engg.split_datetime_col()  # split datetime column
#     df = feat_engg.cal_time_diff()
#     df = feat_engg.categorify_columns()  # label encode the categorical columns
#     df = feat_engg.target_encode_columns()  # target encode the categorical columns
#     df = feat_engg.count_encode_columns()  # count encode the categorical columns
#     # transform to gaussian distribution.
#     df = feat_engg.transforming_target_continuous()
#     df = df[config[required_cols_prediction]]
#     # Make prediction using model loaded from disk as per the data.
#     prediction = model.predict([np.array(list(data.values()))])
#     return {'target': prediction[0]}


# if __name__ == '__main__':
#     uvicorn.run(app, host='0.0.0.0', port=8000)


# @app.route('/api', methods=['POST'])
# class Predict(Resource):
#     def post(self):
#         # Get the data from the POST request.
#         data = request.get_json(force=True)
#         df = data.to_json(orient='records')
#         feat_engg = FeatEngg(df, config, logger)
#         df = feat_engg.split_datetime_col()  # split datetime column
#         df = feat_engg.cal_time_diff()
#         df = feat_engg.categorify_columns()  # label encode the categorical columns
#         df = feat_engg.target_encode_columns()  # target encode the categorical columns
#         df = feat_engg.count_encode_columns()  # count encode the categorical columns
#         # transform to gaussian distribution.
#         df = feat_engg.transforming_target_continuous()
#         df = df[config[required_cols_prediction]]
#         # Make prediction using model loaded from disk as per the data.
#         prediction = model.predict([np.array(list(data.values()))])

#         # Take the first value of prediction
#         output = prediction[0]
#         return output


# if __name__ == '__main__':
#     app.run(port=1002, debug=True)
