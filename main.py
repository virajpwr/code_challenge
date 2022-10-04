
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
    df.to_csv('./data/interim/preprocessed_data.csv', index=False)

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
    df.to_csv('./data/interim/feature_engg_tranformed.csv', index=False)

#---------------------------------------------------------- Feature Selection ----------------------------------------------------------------------------- #
#  Feature selection process for continous variables
    logger = logs(path="logs/", file="feature_selection.logs")
    logger.info(
        "Reading the feature engineered data and performing feature selection")

    # Feature selection using lasso regression

    cont_features = features_selection_continous_var(
        df, config['continous_col_feature_selection'], config['target_col'], config['lasso_params']['params'], logger)
    var_sel_features = cont_features.variance_threshold()  # Variance threshold

    cont_features.var_list = var_sel_features
    print("variance threshold features: ", var_sel_features)
    lasso_var = cont_features.lasso_reg(var_sel_features)
    print("Lasso regression features: ", lasso_var)
    cont_features.var_list = lasso_var
    # df[lasso_var].to_csv('./data/interim/lasso_features.csv', index=False)
    corr_features = cont_features.find_correlated_feature(lasso_var)
    print("Correlated features: ", corr_features)
    final_continous_features = cont_features.mutual_info(
        corr_features, lasso_var)
    print("Mutual information score: ", final_continous_features)

    # Feature selection for categorical variables
    cat_features_selection = features_selection_cat_var(
        df, config['cat_cols_feature_selection'], logger)
    var_sel_cat = cat_features_selection.variance_threshold()
    print("Variance threshold features: ", var_sel_cat)
    cat_features_selection.var_list = var_sel_cat
    final_cat_var = cat_features_selection.perform_anova_test(var_sel_cat)
    print('final cat', final_cat_var)
    selected_features = list(final_continous_features) + list(final_cat_var)
    df[selected_features].to_csv(
        './data/processed/final_features_data.csv', index=False)

# #--------------------------------------------- Model Training ----------------------------------------------------------------------------- #
    # Model training
    logger = logs(path="logs/", file="model_training.logs")
    logger.info(
        "Reading the final data from the data/processed folder and training the model")

    train_model = model_training(
        df, config['target_col'], selected_features, config, logger)  # Instantiate the model training class

    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_model.split_data()
    # log the splits
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")

    # Train baseline model
    logger.info("Training the baseline model")
    theta = train_model.base_model(X_train, y_train)

    # XGB model
    logger.info("Training the XGB model")
    dtest, test_set = train_model.convert_to_DMatrix()
    xgb = train_model.train_xgb()

    # Hyper parameter tuning for random forest
    logger.info("Hyper parameter tuning for random forest")

    # hyperparameter tuning for random forest
    params = train_model.hyperparameter_tuning_randomforest()
    # Train the random forest model on the best parameters
    rf_model = train_model.train_random_forest_from_best_params(
        params, X_train, y_train)

    train_model.save_model()  # save models
# # #------------------------------------------------------ Model Evaluation ----------------------------------------------------------#

    # predict on test data using baseline model
    logger = logs(path="logs/", file="model_evaluation.logs")
    logger.info("Predicting on test data using baseline model")

    pred = model_predictions(theta, X_test, y_test, logger)
    y_pred_reg = pred.predict_linear_reg()  # predict on test data

    # make predictions for test data using random forest trained on best params returned by hyperparameter tuning.
    logger.info("Predicting on test data using random forest")
    pred.model = rf_model  # assign the model to the model attribute
    y_pred_rf = pred.predict_rf_model(rf_model)  # predict on test data
    # feature_imp = pred.rf_feature_importance(rf_model)

    # predict on test data using xgb model
    pred.model = xgb
    pred.predict_xgb(dtest, test_set, xgb)


#----------------------------------------- Create plots ----------------------------------------------------------#
    logger.info("Creating plots")
    plots = visualize(X_test, y_test, y_pred_rf,
                      rf_model, df, selected_features)
    # plots.y_pred = y_pred_reg
    logger.info("Creating plots for baseline model")
    plots.base_model_plots(y_pred_reg)  # Plot for baseline model

    logger.info("Creating plots for random forest model")
    plots.rf_feature_importance()  # Get the feature importance for random forest

    # Plot the learning curve for random forest
    plots.visualize_learning_curve(rf_model)
    plots.y_pred = y_pred_rf
    plots.pred_plot(y_pred_rf)  # plot actual vs predicted for random forest


if __name__ == "__main__":
    main()
