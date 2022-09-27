from imports import *


class features_selection_cont_var(object):
    """_summary_: A class to select the features for the continous variables
    feature selection for continous variables. 
    Steps:
        1. If MI score of all continous variable.
        2. select columns with non zero MI score.
        3. Find correlated features with non zero MI score with coeff >= 0.70.
        4. Remove the correlated feature with low MI score.

        Eg: MI score of feature1 = 0.5, feature2 = 0.4, feature3 = 0.3
        feature1 and feature2 are correlated with coeff = 0.8
        feature1 will be removed as it has low MI score.  

    parameters:
        df {dataframe}: A dataframe with the raw data
        var_list {list}: A list with the continous variables
        target {str}: A string with the target variable

    methods:
        find_correlated_feature(): A function to find the correlated features with threshold 0.70.

        mutual_info(): A function to find the mutual information of the continous variables.

        remove_corr_features(): A function to remove the correlated features with low mutual information score and select the feature with high mutual information score.

    """

    def __init__(self, df, var_list: list, target: str, logger: Logger) -> None:
        self.df = df
        self.var_list = var_list
        self.target = target
        self.logger = logger

    def find_correlated_feature(self, non_zero_mi_cols: list) -> list:
        """
        __Summary__: A function to find the correlated features with threshold 0.70.

        Parameters:
            non_zero_mi_cols {list}: A list with the continous variables with non zero MI score

        returns:
            var_corr {list}: A list with the correlated features with threshold 0.70.
        """
        self.logger.info(
            "Find correlaion in the following columns: {}".format(non_zero_mi_cols))

        # Correlated features
        self.var_list = non_zero_mi_cols
        corrdf = self.df[self.var_list].corr().abs().unstack().sort_values(
        ).drop_duplicates()  # find the correlation between the features
        # convert the series to dataframe
        corrdf = pd.DataFrame(corrdf[:].reset_index())
        corrdf.columns = ['var1', 'var2', 'coeff']  # rename the columns
        # select the features with coeff >= 0.70
        corrdf1 = corrdf[corrdf['coeff'] >= 0.70]
        # remove the duplicate features
        corrdf1 = corrdf1[corrdf1['var1'] != corrdf1['var2']]
        corr_var = corrdf1['var1'].tolist()
        # create a list with the correlated features
        corr_var.extend(corrdf1['var2'].tolist())
        return corr_var

    def mutual_info(self) -> list:
        """
        __Summary__: A function to find the mutual information of the continous variables.

        Parameters:
            None
        returns:
            non_zero_cols {list}: A list with the continous variables with non zero MI score.
            mutual_info{dataframe}: A dataframe with the continous variables and their MI score.
        """
        self.logger.info(
            "Find the mutual information of the continous variables")
        # Mutual information
        # Use selectkbest to find the MI score
        fs = SelectKBest(score_func=mutual_info_regression, k=30)
        rt = fs.fit(self.df[self.var_list],
                    self.df[self.target])  # fit the data
        mutual_info = dict(zip(self.var_list, rt.scores_))
        mutual_info = pd.DataFrame(
            mutual_info.items(), columns=['feature', 'score'])
        # select the features with non zero MI score
        non_zero_cols = mutual_info.loc[mutual_info['score'] != 0].feature.tolist(
        )
        # return the dataframe with MI score and the list with the features with non zero MI score
        return mutual_info, non_zero_cols

    def remove_corr_features(self, mi_df, var_corr) -> list:
        self.logger.info("Remove the correlated features with low MI score")
        """ 
        __Summary__: A function to remove the correlated features with low mutual information score and select the feature with high mutual information score.
        parameters:
            mi_df {dataframe}: A dataframe with the continous variables and their MI score.
            var_corr {list}: A list with the correlated features with threshold 0.70.
        returns:
            final_cols {list}: A list with the selected features.
        """
        self.df = mi_df
        self.var_list = var_corr
        continous_features = self.df.loc[self.df['feature'].isin(
            self.var_list)]  # select the features with MI score
        mi_score_df = self.df[self.df.feature != continous_features.sort_values(
            by='score', ascending=False).iloc[-1, 0]]  # select the feature with high MI score
        final_cont_features = mi_score_df.loc[mi_score_df['score'] != 0].feature.tolist(
        )  # select the features with non zero MI score
        self.logger.info(
            "Selected continous features: {}".format(final_cont_features))
        return final_cont_features


class features_selection_cat_var(object):
    """_summary_: A class to select the features for the categorical variables

    parameters:
        df {dataframe}: A dataframe with the raw data
        var_list {list}: A list with the categorical variables
    methods:
        perform_anova_test(): A function to perform anova test and select the features with p-value < 0.05.
    """

    def __init__(self, df, var_list, logger: Logger) -> None:
        self.df = df
        self.var_list = var_list
        self.logger = logger

    def perform_anova_test(self) -> list:
        """
        __Summary__: A function to perform anova test and select the features with p-value < 0.05.
        Null hypothesis: different categories in a column have means are equal (no variation in means of groups)

        Alternative hypothesis: At least, one category mean is different from other groups.
        This test is used to see if there is a significant difference between the means of two or more groups and select the features with p-value < 0.05.

        parameters:
            None
        returns:
            cat_features{list}: A list with the selected features with 
        """
        self.logger.info(
            "Perform anova test for the following columns: {}".format(self.var_list))
        p_value = [anova(self.df, str(x))
                   for x in self.var_list]  # Loop through the categorical variables and perform anova test
        vars = [i for i in self.var_list]
        df_temp = pd.DataFrame(
            {'feature': vars, 'p_value': p_value})
        # select the features with p-value < 0.05
        cat_features = df_temp[df_temp['p_value'] < 0.05].feature.tolist()
        self.logger.info(
            "Selected categorical features: {}".format(cat_features))
        return cat_features


# class features_selection_cont_var(object):

#     def feature_selection(self):
#         # Feature selection using Lasso
#         X = self.df[self.var_list]
#         y = self.df[self.target]
#         lasso = Lasso(alpha=0.01)
#         lasso.fit(X, y)
#         col_names = X.columns[lasso.coef_ != 0]
#         return self.df[col_names]

    # def feature_selection(self):
    #     model = DecisionTreeRegressor()

    #     rfe = RFECV(estimator=model, step=10, cv=5)

    #     rfe_fit = rfe.fit(self.df[self.var_list], self.df[self.target])

    #     rfe_dict = dict(zip(np.array(self.var_list), rfe_fit.ranking_))

    #     rfe_dict = {k: v for k, v in sorted(
    #         rfe_dict.items(), key=lambda item: item[1])}

    #     col_names = [k for k, v in rfe_dict.items() if v == 1]
    #     return col_names

    # def save_features_after_feature_selection(col_names, config):
    #     pd.Series(col_names).to_csv(os.path.join(
    #         config['PATHS']['Project_path'] + 'data/', 'final_features.csv'))
