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

    def __init__(self, X_test, y_test, y_pred_baseline, y_pred_random_forest, model, df, selected_features) -> None:
        """
        Initialize the class with the required parameters
        parameters:
            - X_test : The test data
            - y_test : The test target
            - y_pred_baseline : The predicted values from the baseline model
            - y_pred_random_forest : The predicted values from the random forest model
            - model : The model
            - df : The dataframe
            - selected_features : The selected features


        """
        self.y_test = y_test
        self.model = model
        self.df = df
        self.selected_features = selected_features
        self.y_pred_random_forest = y_pred_random_forest
        self.y_pred_baseline = y_pred_baseline
        self.X_test = X_test

    def base_model_plots(self):
        """_summary_: This function is used to plot the residual plot, qq plot and actual vs predicted plot for the baseline model 
            and save the plots in eval/plots folder.
        
        parameters:
            y_pred_baseline {np.array}: The predicted values from the baseline model
            y_test {np.array}: The test target
            
        returns:
            None
        """
        # create linear regression plot, residuals plot and histogram of residuals
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Linear Regression Model')
        ax1.scatter(self.y_test, self.y_pred_baseline, color='blue')
        p1 = max(max(self.y_pred_baseline), max(self.y_test))
        p2 = min(min(self.y_pred_baseline), min(self.y_test))
        ax1.plot([p1, p2], [p1, p2], 'b-')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.set_title('Actual vs Predicted')
        ax2.scatter(self.y_pred_baseline, self.y_test-self.y_pred_baseline)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Residual')
        ax2.set_title('Residual Plot')
        ax3.hist(self.y_pred_baseline - self.y_test)
        ax3.set_xlabel('Residual')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Histogram of Residuals')
        plt.savefig('./reports/plots/linear_regression_plots.png')



    def rf_feature_importance(self):
        """_summary_: This function is used to plot the feature importance for the random forest model 
            and save the dataframe with feature importance in reports/eval folder. Save the Feature Importance 
            plot in reports/plots folder.

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


    def pred_plot(self):
        """_summary_: This function is used to plot the actual vs predicted plot for the random forest model.
                        and save the plot in reports/plots folder.
        parameters:
            y_pred_random_forest {np.array}: The predicted values from random forest model

        returns:
            None
        """
        # plot predicted vs actual
        plt.figure(figsize=(10, 10))
        plt.scatter(self.y_test, self.y_pred_random_forest, c='crimson')
        p1 = max(max(self.y_pred_random_forest), max(self.y_test))
        p2 = min(min(self.y_pred_random_forest), min(self.y_test))
        plt.plot([p1, p2], [p1, p2], 'b-')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted for Random Forest')
        plt.savefig('./reports/plots/actual_vs_predicted_random_forest.png')
    
    def actual_fitted_plot(self):
        """_summary_: This function is used to plot the actual vs fitted distribution plot for the random forest model."""
        plt.figure(figsize=(10, 10))
        ax = sns.distplot(self.y_test, hist=False, color="r", label="Actual Value") 
        sns.distplot(self.y_pred_random_forest, hist=False, color="b", label="Fitted Values" , ax=ax)
        plt.legend()
        plt.title('Actual vs Fitted Values for Random Forest')
        plt.savefig('./reports/plots/actual_vs_fitted_random_forest.png')

    def visualize_learning_curve(self):
        """_summary_: This function is used to plot the learning curve for the random forest model.
                        and save the plot in reports/plots folder.

        parameters:
            model {sklearn model}: The random forest model

        returns:
            None

        """
        train_size = np.int64(np.linspace(1, 6000, 5)) # split the training data into 5 parts
        # plot learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.df[self.selected_features], self.df['target'], cv=5,
            scoring='neg_mean_squared_error', n_jobs=-1, train_sizes=train_size)

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
