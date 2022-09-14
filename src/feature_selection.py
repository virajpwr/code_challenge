import matplotlib.pyplot as plt
import numpy as np
from src.utils import load_config
from sklearn.feature_selection import SelectFromModel
from featurewiz import FeatureWiz
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
import yaml
import seaborn as sns
import warnings
import pandas as pd
import os
import sys
root = os.path.abspath('..')
sys.path.append(root)
warnings.filterwarnings('ignore')
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', None)


def feature_selection(df, var_list, target):
    '''
    Feature selection is done in following steps:
    1. Remove highly correlated features r > 0.7.
    2. Remove features with low mutual information score when multiple features are highly correlated.
    3. Recursively do feature selection using XGBoost to find the best features.
    The above steps are done using open source library featurewiz.
    '''
    X_train, y_train = df[var_list], target
    features = FeatureWiz(corr_limit=0.70, feature_engg='', category_encoders='',
                          dask_xgboost_flag=False, nrows=None, verbose=1)
    X_train_selected = features.fit_transform(X_train, y_train)
    ### provides the list of selected features ###
    return features.features

# Function to save features after feature selection


def save_features_after_feature_selection(col_names, config):
    pd.Series(col_names).to_csv(os.path.join(
        config['PATHS']['Project_path'] + 'data/', 'final_features.csv'))

# if __name__ == '__main__':
#     config = load_config('config.yaml')
#     df = pd.read_csv(os.path.join(config['PATHS']['Project_path'] + 'data/', 'train.csv'))
#     df.head()
#     target = df['target']
#     del df['target']
#     selected_features = pd.read_csv(os.path.join(config['PATHS']['Project_path'] + 'data/', 'final_features_before_fs.csv'))
#     selected_features = list(selected_features['0'])
#     selected_features.remove('target')
#     final_selected_features = feature_selection(df, selected_features, target)
#     save_features_after_feature_selection(final_selected_features, config)
#     #pd.Series(final_selected_features).to_csv(os.path.join(config['PATHS']['Project_path'] + 'data/', 'final_features.csv'))
