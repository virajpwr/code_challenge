from fastapi import FastAPI
import uvicorn
from flask import Flask, jsonify, make_response, request
from imports import *

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def score():
    # Load config file
    with open(r".\config.yml", 'r') as f:
        config = yaml.safe_load(f)
    logger = logs(path="logs/", file="model_training.logs")
    logger.info(
        "Reading the final data from the data/processed folder and training the model")
    content = request.json  # get the content of the request
    df = pd.DataFrame(content, index=[0])  # convert the content to a dataframe
    # read the preprocessed data.
    df_processed = pd.read_csv(r'./data/preprocessed_data.csv')
    # append the content data to the preprocessed data for transformation
    df_test = df_processed.append(df, ignore_index=True)
    # Load the model from the models folder
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
    logger.info(
        "Prediction done")
    return {'target': prediction[0]}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
