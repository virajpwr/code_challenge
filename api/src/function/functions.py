from imports import *
from pathlib import Path

# Version of the model
__version__ = '0.1.0'

# choose the base directory
BASE_DIR = Path(__file__).resolve(strict=True).parent.parent.parent

# Load the model
with open(f"{BASE_DIR}/models/randomforest-{__version__}.dat", "rb") as f:
    model = joblib.load(f)


def get_model_response(input:json) -> json:
    """__summary__ = This function takes the input and returns the prediction

    parameters:
        - input {json}: Input data in json format.
    output:
        - prediction {float}: Predicted value
    """
    # Load the config file
    config = load_config(r"./config.yml")
    logger = logs(path="logs/", file="preidctions.logs")

    logger.info('Loading the model')
    # convert json to dataframe
    X = pd.json_normalize(input
    )
    # X = pd.json_normalize(input.__dict__)

    # logger.info('the input is {}'.format(type(X)))

    # Get processed data from preprocessing step in the training pipeline.
    df_processed = pd.read_csv(r'./data/preprocessed_data.csv')

    # Join the input data with the processed data for target encoding
    df_test = df_processed.append(X, ignore_index=True)

    # Perform feature engineering on the input data   
    feat_engg = FeatEngg(df_test, config, logger)

    df_test = feat_engg.split_datetime_col()  # split datetime column

    df_test = feat_engg.cal_time_diff()

    # label encode the categorical columns
    df_test = feat_engg.categorify_columns()

    df_test = feat_engg.target_encode_columns()

    # count encode the categorical columns
    df_test = feat_engg.count_encode_columns()

    # Get the value for pure_seastone column to select the columns for prediction
    purestone = X['pure_seastone'].values[0]

    # Select the row which has the pure_seastone value X value so that it only predicts the value for that row.
    predict_x = df_test.loc[df_test['pure_seastone']
                            == purestone]

    # Make predictions..
    pred = predict_x[config['required_cols_prediction']]

    prediction = model.predict(pred)
    
    # Return predicted value.
    return prediction[0]
