
from imports import *


class model_training(object):
    """__summary__ = "This is a class for training the base model, XGboost model and tuning random forest model and saving the model.

    parameters:
        df{pd.Dataframe}: Cleaned dataframe after preprocessing and feature engineering.
        target{str}: Target column name.
        final_columns{list}: List of final columns after feature selection.
        config{dict}: Dictionary of all the parameters.

    methods:
        convert_to_DMatrix: Convert the dataframe to DMatrix for xgboost and split the data into train, test and validation with 80:10:10 ratio.
        split_data: Split the data into train and test. Ratio of 80:20 for baseline model. 80% of the data is taken for training to reduce variance in the data which causes overfitting.  
        base_model: Train the baseline model and save the model.
        hyperparameter_tuning_randomforest: Hyperparameter tuning for random forest model. Perform randomized search to find the best parameters for the model with 5 fold cross validation.
        train_random_forest_from_best_params: Train the random forest model with best parameters and save the model.
        hyperparameter_tuning_xgb: Function forHyperparameter tuning for xgboost model.
        save_model: Function to Save the model.
    """

    def __init__(self, df, target, final_columns, config, logger: Logger) -> None:
        """
        __summary_: Initialize the class with the required parameters.

        parameters:
            df{pd.Dataframe}: Cleaned dataframe after preprocessing and feature engineering.
            target{str}: Target column name.
            final_columns{list}: List of final columns after feature selection.
            config{dict}: Dictionary of all the parameters.
            logger{Logger}: Logger object.

        """
        self.df = df
        self.target = target
        self.final_columns = final_columns
        self.config = config
        self.logger = logger
        # self.model_params = self.config["logging_params"]["xgb"]
        # self.param_grid = self.config["parameter_grid"]["params"]

    def convert_to_DMatrix(self):
        """_summary_: Convert the dataframe to DMatrix for xgboost and split the data into train, test and validation with 80:10:10 ratio.

        Returns:
            dtest {DMatrix}: DMatrix for test data.
            test {pd.Series}: 10% of the data for test.
        """
        self.logger.info(
            "Converting to DMatrix and splitting the data in 80:10:10 ratio")
        # convert to DMatrix for xgboost which is an internal data structure for xgboost.
        self.train, self.temp = train_test_split(
            self.df, test_size=0.2, random_state=2022)  # 80% training data.
        self.test, self.valid = train_test_split(
            self.temp, test_size=0.5, random_state=2022)  # 10% test data and 10% validation data.
        self.dtrain = xgb.DMatrix(
            data=self.train[self.final_columns], label=self.train.target, enable_categorical=True)  # convert to DMatrix for xgboost and enable categorical features.
        self.dvalid = xgb.DMatrix(
            data=self.valid[self.final_columns], label=self.valid.target, enable_categorical=True)
        self.dtest = xgb.DMatrix(
            data=self.test[self.final_columns], label=self.test.target, enable_categorical=True)
        # save dmatrix into binary buffer
        # self.dtest.save_binary("dtest.dmatrix")
        self.evals = [(self.dtrain, "train"), (self.dvalid, "valid")]
        self.num_round = 500
        return self.dtest, self.test

    def split_data(self):
        """
        summary_: Split the data into train and test. Ratio of 80:20 for baseline model
        Returns:
            X_train {pd.Series}: 80% of the data for training. 
            X_test {pd.Series}: 20% of the data for testing.
            y_train {pd.Series}: 80% of the data for training.
            y_test {pd.Series}: 20% of the data for testing.
        """
        self.logger.info("Splitting the data into train and test")
        X = self.df[self.final_columns]
        y = self.df['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=23)
        X_test.to_csv(
            r'.\data\interim\X_test.csv')
        y_test.to_csv(
            r'.\data\interim\y_test.csv')
        return X_train, X_test, y_train, y_test

    # Multiple regression model is our baseline model
    def base_model(self, X_train: pd.Series, y_train: pd.Series) -> object:
        """
        __summary__: Train the baseline model which is multiple regresssion and save the model.
        parameters:
            X_train {pd.Series}: 80% of the data for training.
            y_train {pd.Series}: 80% of the data for training.
        Returns:
            model {object}: trained multiple regression model.
        """
        self.logger.info("Training the baseline model")
        # Add x0=1 to the first column of X
        X_train_0 = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        # Build the model
        theta = np.matmul(np.linalg.inv(
            np.matmul(X_train_0.T, X_train_0)), np.matmul(X_train_0.T, y_train))
        # save the model.
        pickle.dump(theta, open(os.path.join(
            self.config['PATHS']['Project_path'] + 'models/', self.config['base_model']), "wb"))
        return theta

    def train_xgb(self):
        # train baseline model for comparison.
        self.logger.info("Training the xgboost model")
        """
        __summary__: Train the xgboost model with the parameters set in config file.

        Returns:
            model {object}: trained xgboost model.
        """
        self.xgb_model = xgb.train(
            params=self.config["logging_params"]["xgb"],
            dtrain=self.dtrain,
            num_boost_round=self.num_round,
            evals=self.evals,
        )
        return self.xgb_model

    def hyperparameter_tuning_randomforest(self):
        """
        __summary__: Function for Hyperparameter tuning for random forest model.

        returns: best_params{dict}: Dictionary of best parameters for random forest model.

        """
        self.logger.info("Hyperparameter tuning for random forest model")
        # hyperparameter tuning for random forest.
        self.rf_model = RandomForestRegressor()
        self.random_search = RandomizedSearchCV(
            self.rf_model, self.config["parameter_grid_rf"]["param_grid"], scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=-1)
        self.random_search.fit(self.df[self.final_columns],
                               self.df.target)
        self.best_params = self.random_search.best_params_

        self.logger.info(
            "Best parameters for random forest model: %s", self.random_search.best_params_)
        self.logger.info("Lowest RMSE for random forest model: %s",
                         (-self.random_search.best_score_)**(1/2.0))
        # save the best parameters for random forest model.
        with open(os.path.join(self.config['PATHS']['Project_path'] + 'models/', self.config['best_params_rf']), 'w') as file:
            file.write(json.dumps(self.best_params))
        return self.best_params

    def train_random_forest_from_best_params(self, params, X_train, y_train):
        """
        __summary__: Train the random forest model with the best parameters from hyperparameter tuning.

        params: params{dict}: Dictionary of best parameters for random forest model.
                X_train{pd.Series}: 80% of the data for training.
                y_train{pd.Series}: 80% of the data for training.

        returns: rf_model{object}: Trained random forest model.
        """
        self.logger.info("Training the random forest model")
        self.logger.info("Best parameters for random forest model: %s", params)
        # Get the best parameters from the grid search and train the model
        self.randomforest = RandomForestRegressor(**params, oob_score=True)
        self.randomforest.set_params(**params)
        self.randomforest.fit(X_train, y_train)
        joblib.dump(self.randomforest, open(os.path.join(
            self.config['PATHS']['Project_path'] + 'models/', self.config['model_name_random_forest']), "wb"), compress=3)  # compress the model to reduce the size of the model
        return self.randomforest

    def save_model(self):
        """
        __summary__: Save the model to the models folder.

        returns:
            model{object}: Trained xgboost model.
        """
        self.logger.info("Saving the model")
        # save the model.
        pickle.dump(self.xgb_model, open(os.path.join(
            self.config['PATHS']['Project_path'] + 'models/', self.config['model_name']), "wb"))
