from imports import *


class visualize(object):
    """_summary_: This function is used to plot the learning curve for the model, feature importance and actual vs predicted plots, residual plot and historgram of residuals for the baseline model and 
    save them in the reports/plots folder. It also saves the feature importance in the reports/eval folder.

    methods: 
        - base_model_plots : This function is used to plot the residual plot, qq plot and actual vs predicted plot for the baseline model.
        - rf_feature_importance : This function is used to plot the feature importance for the random forest model.
        - pred_plot : This function is used to plot the actual vs predicted plot for the random forest model.
        - visualize_learning_curve : This function is used to plot the learning curve for the random forest model.


    """

    def __init__(self, X_test, y_test, y_pred, model, df, selected_features) -> None:
        """
        Initialize the class with the required parameters
        parameters:
            - X_test : The test data
            - y_test : The test target
            - y_pred : The predicted values
            - model : The model
            - df : The dataframe
            - selected_features : The selected features


        """
        self.y_test = y_test
        self.model = model
        self.df = df
        self.selected_features = selected_features
        self.y_pred = y_pred
        self.X_test = X_test

    def base_model_plots(self, y_pred_reg):
        """_summary_: This function is used to plot the residual plot, qq plot and actual vs predicted plot for the baseline model.

        parameters:
            y_pred_reg {np.array}: The predicted values from baseline model
        """
        self.y_pred = y_pred_reg
        # create linear regression plot, residuals plot and histogram of residuals
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Linear Regression Model')
        ax1.scatter(self.y_test, self.y_pred)
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Actual vs Predicted')
        ax2.scatter(self.y_pred, self.y_test-self.y_pred)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Residual')
        ax2.set_title('Residual Plot')
        ax3.hist(self.y_pred - self.y_test)
        ax3.set_xlabel('Residual')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Histogram of Residuals')
        plt.savefig('./reports/plots/linear_regression_plots.png')

        # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # # Scatter plot of actual vs predicted
        # ax[0].scatter(self.y_test, self.y_pred)
        # ax[0].set_xlabel('Actual')
        # ax[0].set_ylabel('Predicted')
        # ax[0].set_title('Actual vs Predicted')
        # # Scatter plot of residuals
        # ax[1].scatter(self.y_pred,  self.y_test - self.y_pred)
        # ax[1].set_xlabel('Predicted')
        # ax[1].set_ylabel('Residuals')
        # ax[2].set_title('Residuals vs Predicted')
        # # qq plot
        # # qq plot of residuals to check normality.
        # stats.probplot(self.y_test - self.y_pred, dist="norm", plot=ax[2])
        # ax[2].set_title('QQ plot of residuals')
        # plt.savefig('./reports/plots/linear_regression_plots.png')

        # ## histogram of residuals
        # plt.figure(figsize=(10, 10))
        # ax[3].hist(self.y_test - self.y_pred)
        # ax[3].xlabel('Residuals')

    def rf_feature_importance(self):
        """_summary_: This function is used to plot the feature importance for the random forest model.

        Returns:
            feature_importance {Dataframe}: Dtaframe containing the feature importance values and the corresponding feature names
        """
        feature_importance = pd.DataFrame(
            {'feature': self.X_test.columns, 'importance': self.model.feature_importances_})
        feature_importance.sort_values(
            'importance', ascending=False, inplace=True)
        feature_importance.to_csv(
            './reports/eval/rf_feature_importance.csv', index=False)
        # plot feature importance
        plt.figure(figsize=(10, 10))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig('./reports/plots/rf_feature_importance.png')
        return feature_importance

    def pred_plot(self, y_pred):
        """_summary_: This function is used to plot the actual vs predicted plot for the random forest model.
        parameters:
            y_pred {np.array}: The predicted values from random forest model

        returns:
            None
        """
        self.y_pred = y_pred
        # plot predicted vs actual
        plt.figure(figsize=(10, 10))
        plt.scatter(self.y_test, self.y_pred)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted for Random Forest')
        plt.savefig('./reports/plots/actual_vs_predicted_random_forest.png')

    def visualize_learning_curve(self, model):
        """_summary_: This function is used to plot the learning curve for the random forest model.

        parameters:
            model {sklearn model}: The random forest model

        returns:
            None

        """
        self.model = model
        # plot learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.df[self.selected_features], self.df['target'], cv=5,
            scoring='neg_mean_squared_error', n_jobs=-1, train_sizes=np.int64(np.linspace(1, 6000, 5)))

        train_scores_mean = -train_scores.mean(axis=1)
        test_scores_mean = -test_scores.mean(axis=1)
        plt.figure(figsize=(10, 10))
        plt.plot(train_sizes, train_scores_mean, label='Training error')
        plt.plot(train_sizes, test_scores_mean, label='Validation error')
        plt.ylabel('MSE', fontsize=14)
        plt.xlabel('Training set size', fontsize=14)
        plt.title('Learning curves for a random forest model',
                  fontsize=18, y=1.03)
        plt.legend()
        plt.savefig('./reports/plots/learning_curve.png')
