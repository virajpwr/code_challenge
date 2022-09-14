import xgboost as xgb
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import joblib
from src.train import model_training
from src.feature_selection import feature_selection, save_features_after_feature_selection
from src.feature_engg import FeatEngg
from src.preprocessing import Preprocessing
from src.utils import load_config
from src.data import dataFrame
# Import necessary modules
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, r2_score
import xgboost as xgb
from pycaret.regression import *
from src.utils import load_config
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
import yaml
import seaborn as sns
import warnings
import pandas as pd
import os
import sys
import joblib
import shap
from PIL import Image
#---------------------------------#
# Page layout
# Page expands to full width
# hide_st_style = """
#             <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style>
#             """
# st.set_page_config(page_title='The Machine Learning App',
#                    layout='wide')
# st.markdown(hide_st_style, unsafe_allow_html=True)
#---------------------------------#
# input features


def user_input_features(X):
    groups_Categorify = st.sidebar.slider('groups_Categorify', 2, 76, 2, 1)
    NA_argon = st.sidebar.slider('NA_argon', 0, 1, 0, 1)
    ts_month = st.sidebar.slider('ts_month', 1, 12, 2, 1)
    TE_groups_Categorify = st.sidebar.slider('TE_groups_Categorify', float(X.TE_groups_Categorify.min(
    )), float(X.TE_groups_Categorify.max()), float(X.TE_groups_Categorify.mean()))

    ts_day = st.sidebar.slider('ts_day', 1, 31, 2, 1)

    crystal_supergroup_Categorify = st.sidebar.slider(
        'crystal_supergroup_Categorify', 2, 3, 2, 1)

    CE_TE_groups_Categorify = st.sidebar.slider('CE_TE_groups_Categorify', float(X.CE_TE_groups_Categorify.min(
    )), float(X.CE_TE_groups_Categorify.max()), float(X.CE_TE_groups_Categorify.mean()))

    ts_weekday = st.sidebar.slider('ts_weekday', 1, 31, 2, 1)

    argon = st.sidebar.slider('argon', float(X.argon.min(
    )), float(X.argon.max()), float(X.argon.mean()))
    place = st.sidebar.slider('place', 1, 2, 0, 1)
    NA_Unnamed_17 = st.sidebar.slider('NA_Unnamed_17', 0, 1, 0, 1)

    crystal_type_Categorify = st.sidebar.slider('crystal_type_Categorify', float(X.crystal_type_Categorify.min(
    )), float(X.crystal_type_Categorify.max()), float(X.crystal_type_Categorify.mean()))

    super_hero_group_Categorify = st.sidebar.slider(
        'super_hero_group_Categorify', 0, 8, 0, 1)
    TE_crystal_type_Categorify = st.sidebar.slider('TE_crystal_type_Categorify', float(X.TE_crystal_type_Categorify.min(
    )), float(X.TE_crystal_type_Categorify.max()), float(X.TE_crystal_type_Categorify.mean()))

    expected_final_factor_x = st.sidebar.slider('expected_final_factor_x', float(X.expected_final_factor_x.min(
    )), float(X.expected_final_factor_x.max()), float(X.expected_final_factor_x.mean()))

    chemical_x = st.sidebar.slider('chemical_x', float(X.chemical_x.min(
    )), float(X.chemical_x.max()), float(X.chemical_x.mean()))

    first_factor_x = st.sidebar.slider('first_factor_x', float(X.first_factor_x.min(
    )), float(X.first_factor_x.max()), float(X.first_factor_x.mean()))

    NA_chemical_x = st.sidebar.slider('NA_chemical_x', 0, 1, 0, 1)

    data = {'groups_Categorify': groups_Categorify, 'NA_argon': NA_argon, 'ts_month': ts_month, 'TE_groups_Categorify': TE_groups_Categorify,
            'ts_day': ts_day, 'crystal_supergroup_Categorify': crystal_supergroup_Categorify, 'CE_TE_groups_Categorify': CE_TE_groups_Categorify,
            'ts_weekday': ts_weekday, 'argon': argon, 'place': place, 'NA_Unnamed_17': NA_Unnamed_17,
            'crystal_type_Categorify': crystal_type_Categorify, 'super_hero_group_Categorify': super_hero_group_Categorify,
            'TE_crystal_type_Categorify': TE_crystal_type_Categorify, 'expected_final_factor_x': expected_final_factor_x, 'chemical_x': chemical_x,
            'first_factor_x': first_factor_x, 'NA_chemical_x': NA_chemical_x}
    features = pd.DataFrame(data, index=[0])
    return features

#---------------------------------#
# Model building


def build_model(df):
    config = load_config("config.yml")  # Load config file
    # 1. Preprocessing
    preprocessing = Preprocessing(df, config)
    preprocessing.processing_missing_values()
    df = preprocessing.preprocess_date_cols(config["date_cols"])
    df = preprocessing.cal_time_diff(config["date_cols"])

    # 2. Feature engineering
    feat_engg = FeatEngg(df=df, config=config)
    # df = feat_engg.feat_engg_date()
    df = feat_engg.categorify_columns()
    df = feat_engg.target_encode_columns()
    df.head()
    df = feat_engg.count_encode_columns()
    df = feat_engg.transforming_target_continuous()

    # 3. feature selection
    selected_features = feature_selection(
        df, config["required_columns"], df["target"])
    save_features_after_feature_selection(selected_features, config)
    cols_to_drop = ["super_hero_group", "crystal_supergroup"]
    selected_features = [e for e in selected_features if e not in cols_to_drop]
    st.pyplot()
    df[selected_features].to_csv('./data/final_data.csv', index=False)
    st.header('features selected after feature selection process')
    st.write(df[selected_features].columns)

    target = df['target']
    train, temp = train_test_split(df, test_size=0.2, random_state=223)
    st.markdown('**1.2. Data splits**')
    st.write('Training set')
    st.info(train.shape)
    test, valid = train_test_split(temp, test_size=0.5, random_state=223)
    st.write('Test set')
    st.info(test.shape)
    dtrain = xgb.DMatrix(train[selected_features], label=train['target'])
    dvalid = xgb.DMatrix(valid[selected_features], label=valid['target'])
    dtest = xgb.DMatrix(test[selected_features], label=test['target'])
    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    xgb_model = xgb.train(
        params=config["logging_params"]["xgb"], dtrain=dtrain, num_boost_round=500, evals=evals)
    y_pred = xgb_model.predict(dtest)
    st.subheader('2. Model Performance')
    mse = mean_squared_error(test['target'], y_pred)
    st.write('MSE of XGBoost:: ', mse)
    st.write('RMSE of XGBoost: ', np.sqrt(mse))
    st.write('R2 of XGBoost: ', r2_score(test['target'], y_pred))
    st.subheader('3. Model Parameters')
    st.write(config["logging_params"]["xgb"])
    st.subheader('3. Model explanation')
    df_ = pd.read_csv('./data/final_data.csv')
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(df_)
    st.header('Feature Importance')
    plt.title('Feature importance based on SHAP values')
    fig = shap.summary_plot(
        shap_values, df_.columns, plot_type="bar", show=False)
    st.pyplot(fig, bbox_inches='tight')
    y_pred = xgb_model.predict(dtest)
    st.header('test data')
    df_train = df_.iloc[2:3, :]
    df_test = df_.iloc[0:1, -1]
    dtest_ = xgb.DMatrix(data=df_train, label=df_test, enable_categorical=True)
    st.write(df_train)
    st.header('prediction')
    xgb_model_load = joblib.load(
        r'D:\code_challenge\code_challenge\models\xgboost.dat')
    y_prediction = xgb_model_load.predict(dtest_)
    st.write(y_prediction)


#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader(
        "Upload your input parquet file", type=["parquet"])

#---------------------------------#
# Main panel
if __name__ == '__main__':
    # Displays the dataset
    st.subheader('1. Dataset')

    if uploaded_file is not None:
        if st.button('train the model'):
            df = pd.read_parquet(uploaded_file)
            st.markdown('**1.1. Glimpse of dataset**')
            st.write(df)
            build_model(df)

            image = Image.open(r'D:\code_challenge\code_challenge\flow.jpeg')
            st.image(image, caption='Sunrise by the mountains')
