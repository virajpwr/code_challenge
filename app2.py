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

#---------------------------------#
# Page layout
# Page expands to full width
hide_st_style = """
            <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style>
            """
st.set_page_config(page_title='The Machine Learning App',
                   layout='wide')
st.markdown(hide_st_style, unsafe_allow_html=True)
#---------------------------------#
# input features

st.subheader('2. Prediction')
X = pd.read_csv('final_data.csv')

# df_train = df_t[['groups_Categorify', 'NA_argon', 'ts_month', 'TE_groups_Categorify',
#                     'ts_day', 'crystal_supergroup_Categorify', 'CE_TE_groups_Categorify',
#                     'ts_weekday', 'argon', 'place', 'NA_Unnamed_17',
#                     'crystal_type_Categorify', 'super_hero_group_Categorify',
#                     'TE_crystal_type_Categorify', 'expected_final_factor_x', 'chemical_x',
#                     'first_factor_x', 'NA_chemical_x']]
y = X.iloc[0:1, -1]
# dtest_ = xgb.DMatrix(data=X, label=y, enable_categorical=True)


def user_input_features():
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


df_t = user_input_features()
xgb_model = joblib.load(
    r'D:\code_challenge\code_challenge\models\xgboost.dat')
# y_prediction = xgb_model.predict(df_t)
# st.write(y_prediction)
