
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

    def split_data(self):
        """
        summary_: Split the data into train and test. Ratio of 80:20 for baseline model
        Returns:
            X_train {pd.Series}: 80% of the data for training.
            X_test {pd.Series}: 20% of the data for testing.
            y_train {pd.Series}: 80% of the data for training.
            y_test {pd.Series}: 20% of the data for testing.
            X_valid {pd.Series}: 20% of the data for validation.
            y_valid {pd.Series}: 20% of the data for validation.
        """


        self.logger.info("Splitting the data into train and test")
        self.X = self.df[self.final_columns]

        self.y = self.df['target']
        self.X_train_first, self.X_test, self.y_train_first, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=2022)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train_first, self.y_train_first, test_size=0.2, random_state=2022)

        # Save the data into parquet files
        
        self.X_test.to_parquet("./data/processed/X_test.parquet")

        self.y_test.to_csv("./data/processed/y_test.csv")
        X_test = self.X_test
        y_test = self.y_test
        return X_test, y_test
    
    # Multiple regression model is our baseline model
    def base_model(self) -> object:
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
        self.X_train_0 = np.c_[np.ones((self.X_train.shape[0], 1)), self.X_train]
        # Build the model
        theta = np.matmul(np.linalg.inv(
            np.matmul(self.X_train_0.T, self.X_train_0)), np.matmul(self.X_train_0.T, self.y_train))
        # save the model.
        pickle.dump(theta, open(os.path.join(
            self.config['PATHS']['Project_path'] + 'models/', self.config['base_model']), "wb"))
        return theta


    def xgb_hyperparameters(self)-> None:
        """
        __summary__: Hyperparameter tuning for xgboost model. 
        Perform randomized search to find the best parameters for the model with 5 fold cross validation.
        
        parameters:
            X_train {pd.Series}: 80% of the data for training.
            y_train {pd.Series}: 80% of the data for training.
        returns:    
            best_params {dict}: Dictionary of best parameters for the model.
        """
        self.xgb = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, eval_metric=["rmse"])

        self.xgb_search = RandomizedSearchCV(self.xgb , param_distributions=self.config["param_grid"]["xgb"], random_state=42,
             n_iter=4, cv=5, verbose=1, n_jobs=1, return_train_score=True, scoring="neg_root_mean_squared_error")
        self.xgb_search.fit(self.X_train, self.y_train)
        self.logger.info("Best parameters for XGBoost: {}".format(self.xgb_search.best_params_))
        self.logger.info("Best score for XGBoost: {}".format(self.xgb_search.best_score_))
        self.logger.info("Best estimator for XGBoost: {}".format(self.xgb_search.best_estimator_))
        self.best_params = self.xgb_search.best_params_


    def train_xgb(self) -> object:
        """
        __summary__: Train the xgboost model on the best parameters and save the model.
        parameters:
            X_train {pd.Series}: 80% of the data for training.
            y_train {pd.Series}: 80% of the data for training.
            X_valid {pd.Series}: 20% of the data for validation from train set.
            y_valid {pd.Series}: 20% of the data for validation from train set.
            
        returns: trained model {object}: trained xgboost model.
        """
        eval_set = [(self.X_train, self.y_train), (self.X_valid, self.y_valid)]
        self.logger.info("Training the XGBoost model")
        xgb_model = xgb.XGBRegressor(**self.best_params, random_state=42)
        xgb_model.fit(self.X_train, self.y_train, eval_metric=["rmse"],
                     eval_set=eval_set, verbose=True, early_stopping_rounds=20)
        pickle.dump(xgb_model, open(os.path.join(
            self.config['PATHS']['Project_path'] + 'models/', self.config['model_name']), "wb"))
        self.logger.info("XGB model Model saved")
        return xgb_model

    def hyperparameter_tuning_randomforest(self) -> None:
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


    def train_random_forest(self) -> object:
        """
        __summary__: Train the random forest model with the best parameters from hyperparameter tuning.

        params: self.best_params{dict}: Dictionary of best parameters for random forest model.
                self.X_train{pd.Series}: 80% of the data for training.
                self.y_train{pd.Series}: 80% of the data for training.

        returns: rf_model{object}: Trained random forest model.
        """
        self.logger.info("Training the random forest model")
        self.logger.info("Best parameters for random forest model: %s", self.best_params)
        # Get the best parameters from the grid search and train the model
        randomforest = RandomForestRegressor(**self.best_params, oob_score=True)
        randomforest.set_params(**self.best_params)
        randomforest.fit(self.X_train, self.y_train)
        joblib.dump(randomforest, open(os.path.join(
            self.config['PATHS']['Project_path'] + 'models/', self.config['model_name_random_forest']), "wb"), compress=3)  # compress the model to reduce the size of the model
        self.logger.info("Random forest model saved")
        return randomforest
