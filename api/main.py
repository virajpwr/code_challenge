from datetime import timedelta
from imports import *
from pydantic import BaseModel
import dataclasses

app = FastAPI()
model_name = 'RandomForest'
version = '1.0.0'


# Not used
# class Features(BaseModel):
#     groups: int
#     target: int
#     when: object
#     super_hero_group: object
#     tracking: int
#     place: int
#     tracking_times: int
#     crystal_type: object
#     unnamed_7: int
#     human_behavior_report: int
#     human_measure: int
#     crystal_weight: float
#     expected_factor_x: int
#     previous_factor_x: float
#     first_factor_x: float
#     expected_final_factor_x: float
#     final_factor_x: float
#     previous_adamantium: float
#     etherium_before_start: int
#     expected_start: object
#     start_process: object
#     start_subprocess1: object
#     start_critical_subprocess1: object
#     predicted_process_end: object
#     process_end: object
#     subprocess1_end: object
#     reported_on_tower: object
#     opened: object
#     chemical_x: float
#     raw_kryptonite: float
#     argon: float
#     pure_seastone: float
#     crystal_supergroup: object
#     cycle: object

#     class Config:
#         arbitrary_types_allowed = True


class Output(BaseModel):
    # Ouput for data validation
    prediction: float


@app.get('/info')
async def model_info():
    """Returns the version of the model and the name of the model"""
    return {
        "name": model_name,
        "version": version
    }


@app.get('/health')
async def service_health():
    """Return service health"""
    return {
        "ok"
    }


@app.post("/predict", response_model=Output)
async def predict_target(data: dict) -> dict:
    '''
    __summary__ = Predicts the target value for the given input
    parameters:
        - data: Input data in json format.
    output:
        - prediction: Predicted value
    '''
    df = pd.json_normalize(data)
    """Returns the prediction for the input data"""
    prediction = get_model_response(data)
    return {
        "prediction": prediction
    }
