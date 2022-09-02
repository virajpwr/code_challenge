# Import necessary modules
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
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
root = os.path.abspath('..')
sys.path.append(root)
warnings.filterwarnings('ignore')
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', None)


class model_training(object):
    def __init__(self, df, target, final_columns, config) -> None:
        self.df = df
        self.target = target
        self.final_columns = final_columns
        self.config = config
        self.model_params = self.config["logging_params"]["xgb"]

    def convert_to_DMatrix(self):
        # convert to DMatrix for xgboost which is an internal data structure for xgboost.
        self.train, self.temp = train_test_split(
            self.df, test_size=0.1, random_state=2022)
        self.test, self.valid = train_test_split(
            self.temp, test_size=0.5, random_state=2022)
        self.dtrain = xgb.DMatrix(
            data=self.train[self.final_columns], label=self.train.target, enable_categorical=True)
        self.dvalid = xgb.DMatrix(
            data=self.valid[self.final_columns], label=self.valid.target, enable_categorical=True)
        self.dtest = xgb.DMatrix(
            data=self.test[self.final_columns], label=self.test.target, enable_categorical=True)
        self.evals = [(self.dtrain, "train"), (self.dvalid, "valid")]
        self.num_round = 500

    def train_baseline_model(self):
        # train baseline model for comparison.
        self.xgb_model = xgb.train(
            params=self.model_params,
            dtrain=self.dtrain,
            num_boost_round=self.num_round,
            evals=self.evals,
        )

    def evaluate_model(self):
        # evaluate the model.
        self.y_pred = self.xgb_model.predict(self.dtest)
        mse = mean_squared_error(self.test.target, self.y_pred)
        print('Root mean squared error', np.sqrt(mse))
        with open('./eval/RMSE.txt', 'w') as f:
            f.write(str(np.sqrt(mse)))
        return self.y_pred

    def hyperparameter_search(self, params):
        # hyperparameter search for xgboost.
        self.xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42,
                                          early_stopping_rounds=5)
        search = RandomizedSearchCV(self.xgb_model,
                                    param_distributions=params,
                                    random_state=42, n_iter=200,
                                    cv=3, verbose=1, n_jobs=1,
                                    return_train_score=True)

        search.fit(self.df[self.final_columns], self.target)

        self.report_best_scores(search.cv_results_, 1)

    def save_model(self):
        # save the model.
        pickle.dump(self.xgb_model, open(os.path.join(
            self.config['PATHS']['Project_path'] + 'models/', self.config['model_name']), "wb"))

    def load_model(self, path):
        # load the model.
        self.xgb_model = pickle.load(open(path, "rb"))
