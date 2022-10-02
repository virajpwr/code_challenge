# from model import SentimentModel, SentimentQueryModel
from fastapi import FastAPI
import uvicorn
from flask import Flask, jsonify, make_response, request
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel
from imports import *

app = FastAPI()


class Features(BaseModel):
    groups: int
    target: int 
    when: object
    super_hero_group: object
    tracking: int 
    place: int 
    tracking_times: int 
    crystal_type: object
    unnamed_7: int 
    human_behavior_report: int 
    human_measure: int 
    crystal_weight: float 
    expected_factor_x: int 
    previous_factor_x: float 
    first_factor_x: float 
    expected_final_factor_x: float 
    final_factor_x: float 
    previous_adamantium: float 
    etherium_before_start: int 
    expected_start: object
    start_process: object
    start_subprocess1: object
    start_critical_subprocess1: object
    predicted_process_end: object
    process_end: object
    subprocess1_end: object
    reported_on_tower: object
    opened: object
    chemical_x: float 
    raw_kryptonite: float 
    argon: float 
    pure_seastone: float 
    crystal_supergroup: object
    cycle: object


@app.post("/predict")
async def predict_target(data: Features):
    with open(r".\config.yml", 'r') as f:
        config = yaml.safe_load(f)
    logger = logs(path="logs/", file="model_training.logs")
    logger.info(
        "Reading the final data from the data/processed folder and training the model")

    data = data.dict()

    data = pd.DataFrame(data, index=[0])

    df_processed = pd.read_csv(r'./data/preprocessed_data.csv')

    df_test = df_processed.append(data, ignore_index=True)

    model = joblib.load(r'./models/rf.dat')

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
    prediction = model.predict(df_test)
    return {
        "prediction": prediction[0]
    }

