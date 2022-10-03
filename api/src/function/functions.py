from imports import *

# with open(r".\config.yml", 'r') as f:
#     config = yaml.safe_load(f)


def get_model_response(input):

<<<<<<< HEAD
    config = load_config(r".\config.yml")
=======
    config = load_config(r"./config.yml")
>>>>>>> v1.2
    logger = logs(path="logs/", file="preidctions.logs")

    logger.info('Loading the model')

    model = joblib.load(r'./models/rf.dat')
<<<<<<< HEAD

    X = pd.json_normalize(input.__dict__)

    logger.info('the input is {}'.format(type(X)))
=======
    X =  pd.json_normalize(input)
    # X = pd.json_normalize(input.__dict__)

    # logger.info('the input is {}'.format(type(X)))
>>>>>>> v1.2

    df_processed = pd.read_csv(r'./data/preprocessed_data.csv')

    df_test = df_processed.append(X, ignore_index=True)

    feat_engg = FeatEngg(df_test, config, logger)

    df_test = feat_engg.split_datetime_col()  # split datetime column

    df_test = feat_engg.cal_time_diff()

    # label encode the categorical columns
    df_test = feat_engg.categorify_columns()

    df_test = feat_engg.target_encode_columns()

    # count encode the categorical columns
    df_test = feat_engg.count_encode_columns()

    purestone = X['pure_seastone'].values[0]

    # Select the row which has the pure_seastone value X value so that it only predicts the value for that row. 
    predict_x = df_test.loc[df_test['pure_seastone']
                            == purestone]

    # Make predictions..
    pred = predict_x[config['required_cols_prediction']]

    prediction = model.predict(pred)

    return prediction[0]
