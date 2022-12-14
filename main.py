
from imports import *
import warnings
warnings.filterwarnings("ignore")

## Branch 
def main():
    """_summary_: This is the main function which will call all the functions for preprocessing the data, 
                    feature engineering, feature selection, model training and saving the model.

    parameters:
            None
        Returns:
            model {object}: A trained model

    """
    # set a logger file
    # logger = logs(path="logs/", file="main.logs")

    # Load config file
    config = load_config(r".\config.yml")

# --------------------------------------------------------------------------- Read the data --------------------------------------------------------------------------- #
    # read the raw data and target data from the data/raw folder and merge it and save it in the data/raw folder in parquet format.
    logger = logs(path="logs/", file="data.logs")

    logger.info("Reading the raw data and target data from the data/raw folder")

    data = read_data(logger)  # Instantiate the merge dataset class
    raw_data, target_data = data.read_data()

    logger.info("Merging the raw data and target data")

    # Merge the raw data and target data
    # Instantiate the merge dataset class
    merge_data = merge_dataset(raw_data, target_data, logger)
    df = merge_data.merge_dataset()

# --------------------------------------------------------------------------- Preprocessing the data --------------------------------------------------------------------------- #
#     # preprocessing the data
    logger = logs(path="logs/", file="preprocessing.logs")
    logger.info(
        "Reading the merged data from the data/raw folder and preprocessing it")

    # Instantiate the preprocessing class
    preprocess = preprocessing(df, config, logger)

    df = preprocess.drop_duplicate_columns()  # Drop duplicate columns
    df = preprocess.convert_dtypes()  # Convert the dtypes of the columns
    df = preprocess.replace_values()  # Replace the values in the columns
    # Process the missing values in the columns
    df = preprocess.processing_missing_values()
    df = preprocess.interpolate_datetime()  # Interpolate the datetime column
    df = preprocess.convert_dtypes()  # Convert the dtypes of the columns
    df[config['date_cols']] = df[config['date_cols']].apply(
        pd.to_datetime)  # Convert the date columns to datetime format
    df = preprocess.drop_duplicate_rows()  # Drop the duplicate rows
    df = preprocess.remove_outliers()  # Remove the outliers using the IQR method
    df.to_parquet('./data/interim/preprocessed_data.parquet.gzip', index=False,compression='gzip')

# # ---------------------------------------------------------- Feature Engineering ----------------------------------------------------------------------------- #
#     # Feature engineering
    logger = logs(path="logs/", file="feature_engg.logs")
    logger.info(
        "Reading the preprocessed data from the data/interim folder and performing feature engineering")

    feat_engg = FeatEngg(df, config, logger)
    df = feat_engg.split_datetime_col()  # split datetime column

    # calculate time difference between two datetime columns
    df = feat_engg.cal_time_diff()
    df = feat_engg.categorify_columns()  # label encode the categorical columns
    df = feat_engg.target_encode_columns()  # target encode the categorical columns
    df = feat_engg.count_encode_columns()  # count encode the categorical columns
    # transform to gaussian distribution.
    df = feat_engg.transforming_target_continuous()

    df.to_parquet('./data/interim/feature_engg_tranformed.parquet.gzip', index=False, compression='gzip')

#---------------------------------------------------------- Feature Selection ----------------------------------------------------------------------------- #
#  Feature selection process for continous variables
    logger = logs(path="logs/", file="feature_selection.logs")
    logger.info(
        "Reading the feature engineered data and performing feature selection")

    # Feature selection using lasso regression
    logger.info("Feature selection of continous variables")
    cont_features = features_selection_continous_var(
        df, config['continous_col_feature_selection'], config['target_col'], config['lasso_params']['params'], logger)
    
    logger
    var_sel_features = cont_features.variance_threshold()  # Variance threshold

    cont_features.var_list = var_sel_features
    logger.info("Performing feature selection using lasso regression")
    lasso_var = cont_features.lasso_reg(var_sel_features)
    cont_features.var_list = lasso_var

    logger.info("Find correlated features")
    corr_features = cont_features.find_correlated_feature(lasso_var)
    print("Correlated features: ", corr_features)
    
    logger.info("remove features with low mi score from the highly correlated features")
    final_continous_features = cont_features.mutual_info(
        corr_features, lasso_var)
    print("Mutual information score: ", final_continous_features)

    # Feature selection for categorical variables
    logger.info("Feature selection of categorical variables")
    cat_features_selection = features_selection_cat_var(
        df, config['cat_cols_feature_selection'], logger)
    
    var_sel_cat = cat_features_selection.variance_threshold()
    print("Variance threshold features: ", var_sel_cat)
    cat_features_selection.var_list = var_sel_cat
    final_cat_var = cat_features_selection.perform_anova_test(var_sel_cat)
    print('final cat', final_cat_var)
    selected_features = list(final_continous_features) + list(final_cat_var)

    selected_features = [i for i in selected_features if i not in config['remove_cols']]
    df[selected_features].to_parquet(
        './data/processed/final_features.parquet.gzip', index=False, compression='gzip')

#--------------------------------------------- Model Training ----------------------------------------------------------------------------- #
    # Model training
    logger = logs(path="logs/", file="model_training.logs")
    logger.info(
        "Reading the final data from the data/processed folder and training the model")

    train_model = model_training(
        df, config['target_col'], selected_features, config, logger)  # Instantiate the model training class

    # Split the data into train and test
    X_test, y_test = train_model.split_data()

    # Train baseline model
    logger.info("Training the baseline model")
    base_line_model = train_model.base_model()
    logger.info("Hyperparameter tuning of XGBoost model")
    # Hyperparameter tuning of XGBoost model
    train_model.xgb_hyperparameters()
    # Train the XGBoost model
    logger.info("Training the XGBoost model with best hyperparameters")
    
    xgb_model = train_model.train_xgb()

    # Hyper parameter tuning for random forest
    logger.info("Hyper parameter tuning for random forest")

    # hyperparameter tuning for random forest
    train_model.hyperparameter_tuning_randomforest()
    # Train the random forest model on the best parameters
    random_forest_model = train_model.train_random_forest()

#------------------------------------------------------ Model Evaluation ----------------------------------------------------------#

    # predict on test data using baseline model
    logger = logs(path="logs/", file="model_evaluation.logs")
    logger.info("Predicting on test data using baseline model")


    # # Evaluate trained model and save the evaluation metric in reports/eval folder
    
    logger.info("Evaluating the baseline model")
    # Evaluation of baseline model
    evaluation = model_eval(base_line_model, X_test,y_test, logger )
    y_pred_baseline = evaluation.predict_linear_reg()

    logger.info("Evaluating the XGBoost model")
    # evaluation of random forest model
    evaluation.model = random_forest_model
    y_pred_rf = evaluation.predict_rf_model(random_forest_model)

    logger.info("Evaluating the random forest model")
    # evaluation of xgboost model
    evaluation.model = xgb_model
    y_pred_xgb = evaluation.predict_xgb(xgb_model)

#----------------------------------------- Create plots ----------------------------------------------------------#
    logger.info("Creating plots")
    plots = visualize(X_test, y_test, y_pred_baseline , y_pred_rf, y_pred_xgb, random_forest_model, df, selected_features)
    
    logger.info("Creating plots for baseline model")
    plots.base_model_plots()  # Plot for baseline model

    logger.info("Creating plots for random forest model")
    plots.rf_feature_importance()  # Get the feature importance for random forest

    # Plot the learning curve for random forest
    plots.visualize_learning_curve()
    plots.pred_plot()  # plot actual vs predicted for random forest
    plots.actual_fitted_plot()  # plot actual vs fitted for random forest
    
    plots.model = xgb_model
    logger.info("Creating plots for XGBoost model")
    plots.plot_learning_curve_xgb(xgb_model)
    
    plots.model = random_forest_model
    plots.tree_interpreter(random_forest_model)
if __name__ == "__main__":
    main()
