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
        self.param_grid = self.config["parameter_grid"]["params"]

    def convert_to_DMatrix(self):
        # convert to DMatrix for xgboost which is an internal data structure for xgboost.
        self.train, self.temp = train_test_split(
            self.df, test_size=0.2, random_state=2022)
        self.test, self.valid = train_test_split(
            self.temp, test_size=0.5, random_state=2022)
        self.dtrain = xgb.DMatrix(
            data=self.train[self.final_columns], label=self.train.target, enable_categorical=True)
        self.dvalid = xgb.DMatrix(
            data=self.valid[self.final_columns], label=self.valid.target, enable_categorical=True)
        self.dtest = xgb.DMatrix(
            data=self.test[self.final_columns], label=self.test.target, enable_categorical=True)
        # save dmatrix into binary buffer
        self.dtest.save_binary("dtest.dmatrix")
        self.evals = [(self.dtrain, "train"), (self.dvalid, "valid")]
        self.num_round = 500

    # Multiple regression model is our base model
    def base_model(self):
        X = self.df[self.final_columns]
        y = self.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=23)
        # Add x0=1 to the first column of X
        X_train_0 = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        X_test_0 = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        # Build the model
        theta = np.matmul(np.linalg.inv(
            np.matmul(X_train_0.T, X_train_0)), np.matmul(X_train_0.T, y_train))
        # Make predictions
        y_pred = np.matmul(X_test_0, theta)
        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        # R_square
        sse = np.sum((y_pred - y_test)**2)
        sst = np.sum((y_test - y_test.mean())**2)
        R_square = 1 - (sse/sst)
        with open('./eval/RMSE_multiple_regression.txt', 'w') as f:
            f.write(str(np.sqrt(mse)))
            f.write(str(R_square))
            f.close()
        print('Root mean squared error of multiple regression model', np.sqrt(mse))
        print('r square of multiple regression', R_square)

    def train_xgb(self):
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
        print('Root mean squared error for xgboost', np.sqrt(mse))
        with open('./eval/RMSE_xgb.txt', 'w') as f:
            f.write(str(np.sqrt(mse)))
        return self.y_pred

    def shap(self):
        # explain the model using SHAP values.
        explainer = shap.TreeExplainer(self.xgb_model)
        shap_values = explainer.shap_values(self.df[self.final_columns])
        shap.summary_plot(
            shap_values, self.df[self.final_columns], plot_type="bar")
        shap.summary_plot(
            shap_values, self.df[self.final_columns])
        plt.savefig(os.path.join(
            self.config['PATHS']['Project_path'] + 'plots/', 'shap_plot.png'))
        plt.show()

    def hyperparameter_search(self):
        # hyperparameter search for xgboost.
        self.xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42,
                                          n_jobs=-1)
        search = RandomizedSearchCV(self.xgb_model,
                                    param_distributions=self.config["parameter_grid"]["params"],
                                    random_state=42, n_iter=200,
                                    cv=5, verbose=1, n_jobs=1,
                                    return_train_score=True)

        search.fit(self.df[self.final_columns], self.target)

        # self.report_best_scores(search.cv_results_, 1)
        with open('best_params.txt', 'w') as f:
            f.write(str(search.best_params_))

    def save_model(self):
        # save the model.
        pickle.dump(self.xgb_model, open(os.path.join(
            self.config['PATHS']['Project_path'] + 'models/', self.config['model_name']), "wb"))

    def load_model(self, path):
        # load the model.
        self.xgb_model = pickle.load(open(path, "rb"))
