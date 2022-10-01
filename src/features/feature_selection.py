from imports import *


class features_selection_cat_var(object):
    """_summary_: A class to select the features for the categorical variables

    parameters:
        df {dataframe}: A dataframe with the raw data
        var_list {list}: A list with the categorical variables
    methods:
        perform_anova_test(): A function to perform anova test and select the features with p-value < 0.05.
    """

    def __init__(self, df, var_list, logger: Logger) -> None:
        """
        __Summary__: Initialize the class

        Parameters:
            df {dataframe}: A dataframe with the raw data
            var_list {list}: A list with the categorical variables
        returns:
            None
        """
        self.df = df
        self.var_list = var_list
        self.logger = logger

    def perform_anova_test(self, var_threhold_list: list) -> list:
        """
        __Summary__: A function to perform anova test and select the features with p-value < 0.05.
        Null hypothesis: different categories in a column have means are equal (no variation in means of groups)

        Alternative hypothesis: At least, one category mean is different from other groups.
        This test is used to see if there is a significant difference between the means of two or more groups and select the features with p-value < 0.05.

        parameters:
            var_threhold_list {list}: A list with the categorical variables
        returns:
            cat_features{list}: A list with the selected features with 
        """
        self.var_list = var_threhold_list
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

    def variance_threshold(self):
        """
        __Summary__: A function to select the features with variance > 0.25.
        parameters:
            None
        returns:
            cat_features{list}: A list with the selected features with variance > 0.25
        """

        # select the features with variance > 0.25 i.e. dropping columns that are 75% or more similar.
        var_thr = VarianceThreshold(threshold=0.25)
        var_thr.fit(self.df[self.var_list])
        var_thr.get_support()
        concol = [column for column in self.df[self.var_list].columns
                  if column not in self.df[self.var_list].columns[var_thr.get_support()]]
        self.logger.info(
            "Feature removed due to low variance: {}".format(concol))
        # remove low variance threshold features from var_list
        self.var_list = [x for x in self.var_list if x not in concol]
        return self.var_list


class features_selection_continous_var(features_selection_cat_var):
    """_summary_: A class to select the features for the continous variables
    feature selection for continous variables. Inherit variance_threshold() from features_selection_cat_var class.
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

    def __init__(self, df, var_list: list, target: str, params, logger: Logger) -> None:
        self.df = df
        self.var_list = var_list
        self.target = target
        self.logger = logger
        self.params = params

    def find_correlated_feature(self, lasso_cols: list) -> list:
        """
        __Summary__: A function to find the correlated features with threshold 0.70.

        Parameters:
            lasso_cols {list}: A list with the continous variables selected from lasso regression.

        returns:
            var_corr {list}: A list with the correlated features with threshold 0.70.
        """
        self.logger.info(
            "Find correlaion in the following columns: {}".format(lasso_cols))

        # Correlated features
        self.var_list = lasso_cols
        corrdf = self.df[self.var_list].corr().abs().unstack().sort_values(
        ).drop_duplicates()  # find the correlation between the features
        # convert the series to dataframe
        corrdf = pd.DataFrame(corrdf[:].reset_index())
        corrdf.columns = ['var1', 'var2', 'coeff']  # rename the columns
        # select the features with coeff >= 0.70
        corrdf1 = corrdf[corrdf['coeff'] >= 0.70]
        # remove the duplicate features
        corrdf1 = corrdf1[corrdf1['var1'] != corrdf1['var2']]
        correlated_pair = list(
            zip(corrdf1['var1'].values.tolist(), corrdf1['var2'].values.tolist())) # create a list of tuples with correlated features
        corr_pair_dict = dict(return_dictionary_list(correlated_pair)) # convert the list of tuples to dictionary
        corr_list = find_remove_duplicates(
            corrdf1['var1'].values.tolist()+corrdf1['var2'].values.tolist()) # create a list of correlated features
        print(len(corr_list))
        if len(corr_list) == 0:
            return self.var_list
        else:
            logger.info("Correlated features: {}".format(corr_list))
            return corr_list

    def lasso_reg(self, variance_threshold_cols: list) -> list:
        """
        __Summary__: A function to perform lasso regression with gridsearch cv features to select the best features.

        Parameters:
            variance_threshold_cols {list}: A list with the continous variables selected from variance threshold.
        returns:
            lasso_cols {list}: A list with the selected features from lasso regression.
        """
        self.var_list = variance_threshold_cols
        print('same cols-------------------------------------> ', self.var_list)
        # Feature selection using Lasso
        self.logger.info("Perform feature selection using Lasso")
        lassocv = GridSearchCV(linear_model.Lasso(),
                               param_grid=self.params,
                               cv=5, scoring="neg_root_mean_squared_error").fit(self.df[self.var_list], self.df[self.target])
        importance = np.abs(lassocv.best_estimator_.coef_)
        idx_third = importance.argsort()[-3]
        threshold = importance[idx_third] + 0.01
        idx_features = (-importance).argsort()[:15]
        sel_features = np.array(self.var_list)[idx_features]
        self.logger.info(
            "Selected features using Lasso: {}".format(sel_features))
        return sel_features

    def mutual_info(self, corr_list: list, lasso_col: list) -> list:
        """
        __Summary__: A function to find the mutual information of correlated features with threshold 0.70 from selected features from lasso regression. 
            If there are more than one correlated features, select the feature with high mutual information score. If there are no correlated features,
             select the features from lasso regression. 
        parameters:
            corr_list {list}: A list with the correlated features with threshold 0.70.
            lasso_col {list}: A list with the selected features from lasso regression.
        returns:
            mi_cols {list}: A list with the selected features that has high MI score from correlated features
        """
        # check if corr_col is same as lasso_col. If there are no correlated features, return the features from lasso regression.
        if list(corr_list) == list(lasso_col):
            self.logger.info("Correlated features and Lasso features are same")
            return lasso_col
        else:
            fs = SelectKBest(score_func = mutual_info_regression, k = 20) # find MI score of the correlated features
            fs.fit(self.df[corr_list], self.df[self.target])
            mutual_info = dict(zip(corr_list, fs.scores_)) # convert the series to dictionary of features and MI score
            # The first variable in list has the highest correlation to the target variable 
            sorted_by_mutual_info = [key for (key, val) in sorted(
                mutual_info.items(), key=lambda kv: kv[1], reverse=True)]
            # select the final list of correlated variables
            selected_corr_list = []
            ## making multiple copies of this sorted list since it is iterated many times ####
            orig_sorted = copy.deepcopy(sorted_by_mutual_info)
            copy_sorted = copy.deepcopy(sorted_by_mutual_info)
            # select each variable by the highest mutual info and see what vars are correlated to it
            for each_corr_name in copy_sorted:
                # add the selected var to the selected_corr_list
                selected_corr_list.append(each_corr_name)
                for each_remove in copy_pair[each_corr_name]:
                    # Now remove each variable that is highly correlated to the selected variable
                    if each_remove in copy_sorted:
                        copy_sorted.remove(each_remove)
            # Now combine the uncorrelated list to the selected correlated list above
            rem_col_list = left_subtract(lasso_col, corr_list)
            # list with high MI score
            final_list = rem_col_list + selected_corr_list
            removed_cols = left_subtract(lasso_col, final_list)
            return final_list

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


