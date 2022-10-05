from imports import *


class model_predictions(object):
    """_summary_: This function is used to predict the target values for the test data using basline model and
                random forest model and xgboost model and save evaluation metrics in reports/eval folder.

    parameters:
            model {object}: trained model
            X_test {pd.Series}: test data
            y_test {pd.Series}: test target values
    methods:
            predict_linear_reg: predict the target values for the test data using basline model
            predict_rf_model: predict the target values for the test data using random forest model
            predict_xgb: predict the target values for the test data using xgboost model

    """

    def __init__(self, model, X_test: pd.Series, y_test: pd.Series, logger: Logger) -> None:
        """
        _summary_: This function is used to predict the target values for the test data using basline model and random forest model and xgboost model.

        parameters:
                model {object}: trained model
                X_test {pd.Series}: test data
                y_test {pd.Series}: test target values
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.logger = logger

    def predict_linear_reg(self):
        """
            _summary_: This function is used to predict the target values for the test data using basline model.

        Returns:
            _type_: _description_
        """
        self.logger.info(
            'predicting the target values for the test data using basline model')
        X_test_0 = np.c_[np.ones((self.X_test.shape[0], 1)), self.X_test]
        y_pred_norm = np.matmul(X_test_0, self.model)
        # Make predictions
        y_pred = np.matmul(X_test_0, self.model)
        # Evaluate the model
        mse = np.sum((y_pred_norm - self.y_test)**2) / X_test_0.shape[0]
        rmse = np.sqrt(mse)
        # R_square
        sse = np.sum((y_pred - self.y_test)**2)
        sst = np.sum((self.y_test - self.y_test.mean())**2)
        R_square = 1 - (sse/sst)
        # Calculate VIF to check multicollinearity.
        VIF = 1/(1 - R_square)
        # Mean of residuals
        mean_residuals = np.mean(self.y_test - y_pred)
        # adjusted R_square
        adj_R_square = 1 - (1 - R_square) * \
            (X_test_0.shape[0] - 1) / (X_test_0.shape[0] - X_test_0.shape[1] - 1)
        
        print('VIF of multiple regression', VIF)
        self.logger.info(
            "Root mean squared error of baseline model: {}".format(rmse))

        self.logger.info('VIF of baseline model:{}'.format(VIF))
        self.logger.info(
            'Mean of residuals of baseline model'.format(mean_residuals))

        result = ['rmse', str(rmse),  'VIF', str(
            VIF), 'mean_residuals', str(mean_residuals), 'R_square', str(R_square)]
        # Save the results in reports/eval folder
        with open('./reports/eval/base_model_result.txt', 'w') as f:
            f.write('\n'.join(result))
            f.close()
        return y_pred

    def predict_rf_model(self, rf_model):
        self.logger.info(
            "predicting the target values for the test data using random forest model")
        self.model = rf_model
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        # Evaluate the model
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        # R_square
        R_square = r2_score(self.y_test, y_pred)
        # out of bag score
        oob_error = 1 - self.model.oob_score_

        print('Root mean squared error of random forest model', np.sqrt(mse))
        print('out of bag error', oob_error)
        self.logger.info(
            'Root mean squared error of random forest model:{}'.format(rmse))
        self.logger.info('out of bag score:{}'.format(oob_error))

        # Save the results in reports/eval folder
        results = ['rmse', str(rmse), 'oob_error',
                   str(oob_error), 'R_square', str(R_square)]
        with open('./reports/eval/rf_model_result.txt', 'w') as f:
            f.write('\n'.join(results))
            f.close()
        return y_pred
        

    def predict_xgb(self, dtest, test, xgb_model):
        # evaluate the model.
        self.y_pred = xgb_model.predict(dtest)
        mse = mean_squared_error(test.target, self.y_pred)
        rmse = np.sqrt(mse)
        print('Root mean squared error for xgboost', rmse)
        self.logger.info("Root mean squared error for xgboost:{}".format(rmse))
        with open('./reports/eval/RMSE_xgb.txt', 'w') as f:
            f.write(str(np.sqrt(mse)))
        return self.y_pred
