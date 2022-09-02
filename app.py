import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import joblib
from src.train import model_training
from src.feature_selection import feature_selection, save_features_after_feature_selection
from src.feature_engg import FeatEngg
from src.preprocessing import Preprocessing
from src.utils import load_config
from src.data import dataFrame
st.set_option('deprecation.showPyplotGlobalUse', False)
import xgboost as xgb

### ---- Model building ---####
# def build_model(df):
#     # 1. Preprocessing
#     preprocessing = Preprocessing(df, config)
#     preprocessing.processing_missing_values()
#     df = preprocessing.preprocess_date_cols(config["date_cols"])
#     # 2. Feature engineering
#     feat_engg = FeatEngg(df=df, config=config)
#     # df = feat_engg.feat_engg_date()
#     df = feat_engg.categorify_columns()
#     df = feat_engg.target_encode_columns()
#     df.head()
#     df = feat_engg.count_encode_columns()
#     df = feat_engg.transforming_target_continuous()
#     # 3. feature selection
#     selected_features = feature_selection(
#         df, config["required_columns"], df["target"])
#     save_features_after_feature_selection(selected_features, config)
#     cols_to_drop = ["Cycle", "super_hero_group",
#                     "opened", "crystal_supergroup", "groups", "index"]
#     selected_features = [e for e in selected_features if e not in cols_to_drop]
#     st.markdown('**1.3. Features selected **:')
#     st.write('features')
#     st.info(selected_features)
#     st.write('target')
#     st.info('target')
#     config = load_config("config.yml")
#     target = df['target']
#     train_model = model_training(df, target, selected_features, config)
#     train_model.convert_to_DMatrix()
#     train_model.train_baseline_model()
#     error = train_model.evaluate_model()  # evaluate the model
#     train_model.save_model()
#     return train_model

if __name__ == "__main__":
    st.write("""
    # Datascience coding Challenge
    """)

    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Upload training data'):
        uploaded_file = st.sidebar.file_uploader(
            "upload training data", type=["csv"])
    with st.sidebar.header('2. Upload Model'):
        uploaded_model = st.sidebar.file_uploader(
            "upload model", type=["dat"])

    st.subheader('1. Upload test data')

    if uploaded_file is not None and uploaded_model is not None:
        df = pd.read_csv(uploaded_file)
        model = joblib.load(uploaded_model)
        st.markdown('**1.1. Your data**')
        st.write(df)
        df_ = df.iloc[1:2,:-1]
        st.markdown('**1.2. Test data**')
        st.write(df_)
        ts = xgb.DMatrix(df.iloc[1:2,:-1], label=df['target'])


    #### ------------- ##

    ### -------------- Main ------------- ####
    # if uploaded_file is not None:
    #     df = pd.read_parquet(uploaded_file)
    #     st.markdown('**1.1. Your data**')
    #     st.write(df)
    #     if st.button('train'):
    #         config = load_config("config.yml")
    #         preprocessing = Preprocessing(df, config)
    #         preprocessing.processing_missing_values()
    #         df = preprocessing.preprocess_date_cols(config["date_cols"])
    #         # 2. Feature engineering
    #         feat_engg = FeatEngg(df=df, config=config)
    #         # df = feat_engg.feat_engg_date()
    #         df = feat_engg.categorify_columns()
    #         df = feat_engg.target_encode_columns()
    #         df.head()
    #         df = feat_engg.count_encode_columns()
    #         df = feat_engg.transforming_target_continuous()
    #         # 3. feature selection
    #         selected_features = feature_selection(
    #             df, config["required_columns"], df["target"])
    #         save_features_after_feature_selection(selected_features, config)
    #         cols_to_drop = ["Cycle", "super_hero_group",
    #                         "opened", "crystal_supergroup", "groups", "index"]
    #         selected_features = [
    #             e for e in selected_features if e not in cols_to_drop]
    #         st.markdown('**1.3. Features selected **:')
    #         st.write('features')
    #         st.info(selected_features)
    #         st.write('target')
    #         st.info('target')
    #         config = load_config("config.yml")
    #         target = df['target']
    #         train_model = model_training(df, target, selected_features, config)
    #         train_model.convert_to_DMatrix()
    #         train_model.train_baseline_model()
    #         error = train_model.evaluate_model()  # evaluate the model
    #         st.sidebar.header('Specify Input Parameters')
    #         groups_Categorify = st.sidebar.slider('super_hero_group', float(df.groups_Categorify.min(
    #         )), float(df.groups_Categorify.max()), float(df.groups_Categorify.mean()))

    #         ts_day = st.sidebar.slider('ts_day', 1, 30, 2, 1)

    #         TE_groups_Categorify = st.sidebar.slider('TE_groups_Categorify', float(df.TE_groups_Categorify.min(
    #         )), float(df.TE_groups_Categorify.max()))

    #         ts_month = st.sidebar.slider('ts_month', 1, 12, 2, 1)

    #         ts_weekday = st.sidebar.slider('ts_weekday', 0, 6, 2, 1)

    #         CE_TE_groups_Categorify = st.sidebar.slider('CE_TE_groups_Categorify', float(df.CE_TE_groups_Categorify.min(
    #         )), float(df.CE_TE_groups_Categorify.max()))

    #         crystal_type_Categorify = st.sidebar.slider('crystal_type_Categorify', float(df.crystal_type_Categorify.min(
    #         )), float(df.crystal_type_Categorify.max()))

    #         NA_opened = st.sidebar.slider('NA_opened', float(df.NA_opened.min(
    #         )), float(df.NA_opened.max()), float(df.NA_opened.mean()))

    #         super_hero_group_Categorify = st.sidebar.slider(
    #             'super_hero_group_Categorify', 0, 8, 2, 1)

    #         NA_final_factor_x = st.sidebar.slider(
    #             'NA_final_factor_x', 0, 1, 0, 1)

# if uploaded_file is not None:
#     df = pd.read_parquet(uploaded_file)
#     st.markdown('**1.1. Your data**')
#     st.write(df)
#     config = load_config("config.yml")  # Load config file
#     df = pd.read_parquet("./data/merged_data.parquet")  # Load merged data

#     # 1. Preprocessing
#     preprocessing = Preprocessing(df, config)
#     preprocessing.processing_missing_values()
#     df = preprocessing.preprocess_date_cols(config["date_cols"])

#     # 2. Feature engineering
#     feat_engg = FeatEngg(df=df, config=config)
#     # df = feat_engg.feat_engg_date()
#     df = feat_engg.categorify_columns()
#     df = feat_engg.target_encode_columns()
#     df.head()
#     df = feat_engg.count_encode_columns()
#     df = feat_engg.transforming_target_continuous()

#     # 3. feature selection
#     selected_features = feature_selection(
#         df, config["required_columns"], df["target"])
#     save_features_after_feature_selection(selected_features, config)
#     cols_to_drop = ["Cycle", "super_hero_group",
#                     "opened", "crystal_supergroup", "groups", "index"]
#     selected_features = [e for e in selected_features if e not in cols_to_drop]
#     st.markdown('**1.3. Features selected **:')
#     st.write('features')
#     st.info(selected_features)
#     st.write('target')
#     st.info('target')

#     if st.button('train model'):
#         target = df['target']
#         train_model = model_training(df, target, selected_features, config)
#         train_model.convert_to_DMatrix()
#         train_model.train_baseline_model()
#         error = train_model.evaluate_model()  # evaluate the model
#         train_model.save_model()

#     # 'tracking': tracking,
#     data = {'groups_Categorify': groups_Categorify,
#             'ts_day': ts_day, 'TE_groups_Categorify': TE_groups_Categorify,
#             'ts_month': ts_month, 'ts_weekday': ts_weekday, 'CE_TE_groups_Categorify': CE_TE_groups_Categorify,
#             'crystal_type_Categorify': crystal_type_Categorify, 'NA_opened': NA_opened,
#             'super_hero_group_Categorify': super_hero_group_Categorify, 'NA_final_factor_x': NA_final_factor_x}

#     features = pd.DataFrame(data, index=[0])
# if st.button('train model'):

    # if st.button('train model'):
    #     train_x, test_x, train_y, test_y = get_features(df)
    #     model = train.model(train_x, train_y, k_folds)
    #     st.info(k_folds)
    #     model = model.run()
    #     st.markdown('**1.4 Models performance**:')
    #     model_trained = evaluate.evaluate(model, test_x, test_y)
    #     RMSE, MSE, r_squared = model_trained.evaluate()
    #     st.write('RMSE')
    #     st.info(RMSE)
    #     st.write('MSE')
    #     st.info(MSE)
    #     st.write('r_squared')
    #     st.info(r_squared)
    #     st.success('model trained')
    #     if st.button('predict'):
    #         model = joblib.load('rf_tuned.pkl')
    #         scales = joblib.load('std_scaler.bin')
    #         test_x_scaled = scales.transform(features)
    #         predict = model.predict(test_x_scaled)
    #         st.header('Prediction of Median Value of House (MEDV)')
    #         st.write(predict)
    #         st.write('---')

    # def user_input_features():
    #     super_hero_group = st.sidebar.slider('super_hero_group', float(X.super_hero_group.min(
    #     )), float(X.super_hero_group.max()), float(X.super_hero_group.mean()))
    #     tracking = st.sidebar.slider('tracking', float(tracking.ZN.min()), float(
    #         tracking.ZN.max()), float(tracking.ZN.mean()))
    #     place = st.sidebar.slider('place', float(X.place.min()), float(
    #         X.place.max()), float(X.place.mean()))
    #     tracking_times = st.sidebar.slider('tracking_times', float(X.tracking_times.min(
    #     )), float(X.tracking_times.max()), float(X.tracking_times.mean()))
    #     some_unknown_column = st.sidebar.slider('some_unknown_column', float(X.some_unknown_column.min(
    #     )), float(X.some_unknown_column.max()), float(X.some_unknown_column.mean()))
    #     human_behavior_report = st.sidebar.slider('RM', float(X.human_behavior_report.min(
    #     )), float(X.human_behavior_report.max()), float(X.human_behavior_report.mean()))
    #     human_measure = st.sidebar.slider('human_measure', float(
    #         X.human_measure.min()), float(X.human_measure.max()), float(X.human_measure.mean()))
    #     crystal_weight = st.sidebar.slider('crystal_weight', float(X.crystal_weight.min(
    #     )), float(X.crystal_weight.max()), float(X.crystal_weight.mean()))
    #     expected_factor_x = st.sidebar.slider('expected_factor_x', float(X.expected_factor_x.min(
    #     )), float(X.expected_factor_x.max()), float(X.expected_factor_x.mean()))
    #     first_factor_x = st.sidebar.slider('first_factor_x', float(X.first_factor_x.min(
    #     )), float(X.first_factor_x.max()), float(X.first_factor_x.mean()))
    #     final_factor_x = st.sidebar.slider('final_factor_x', float(X.final_factor_x.min(
    #     )), float(X.final_factor_x.max()), float(X.final_factor_x.mean()))
    #     etherium_before_start = st.sidebar.slider('etherium_before_start', float(X.etherium_before_start.min(
    #     )), float(X.etherium_before_start.max()), float(X.etherium_before_start.mean()))
    #     chemical_x = st.sidebar.slider('chemical_x', float(X.chemical_x.min()), float(
    #         X.chemical_x.max()), float(X.chemical_x.mean()))
    #     raw_kryptonite = st.sidebar.slider('raw_kryptonite', float(X.raw_kryptonite.min(
    #     )), float(X.raw_kryptonite.max()), float(X.raw_kryptonite.mean()))
    #     pure_seastone = st.sidebar.slider('pure_seastone', float(
    #         X.pure_seastone.min()), float(X.pure_seastone.max()), float(X.pure_seastone.mean()))
    #     crystal_supergroup = st.sidebar.slider('crystal_supergroup', float(X.crystal_supergroup.min(
    #     )), float(X.crystal_supergroup.max()), float(X.crystal_supergroup.mean()))
    #     crystal_group_type = st.sidebar.slider('crystal_group_type', float(X.crystal_group_type.min(
    #     )), float(X.crystal_group_type.max()), float(X.crystal_group_type.mean()))
    #     data = {'super_hero_group': super_hero_group, 'tracking': tracking, 'place': place,
    #             'tracking_times': tracking_times,
    #             'some_unknown_column': some_unknown_column,
    #             'human_behavior_report': human_behavior_report,
    #             'human_measure': human_measure,
    #             'crystal_weight': crystal_weight,
    #             'expected_factor_x': expected_factor_x,
    #             'first_factor_x': first_factor_x,
    #             'final_factor_x': final_factor_x,
    #             'etherium_before_start': etherium_before_start,
    #             'chemical_x': chemical_x,
    #             'raw_kryptonite': raw_kryptonite,
    #             'pure_seastone': pure_seastone,
    #             'crystal_supergroup': crystal_supergroup,
    #             'crystal_group_type': crystal_group_type}
    #     features = pd.DataFrame(data, index=[0])
    #     return features
    # if st.button('Show SHAP Graphs'):
    #     model_trained = joblib.load('rf_tuned.pkl')
    #     train_x, test_x, train_y, test_y = get_features(df)
    #     explainer = shap.TreeExplainer(model_trained)
    #     shap_values = explainer.shap_values(train_x)
    #     st.header('Feature Importance')
    #     plt.title('Feature importance based on SHAP values')
    #     shap.summary_plot(shap_values, train_x)
    #     st.pyplot(bbox_inches='tight')
    #     st.write('---')
    #     plt.title('Feature importance based on SHAP values')
    #     shap.summary_plot(shap_values, train_x, plot_type="bar")
    #     st.pyplot(bbox_inches='tight')

    # st.subheader('2. Upload test data')
    # st.markdown('**1.2. Data splits**')

    # st.write('Training set')
    # split
    # st.info(X_train.shape)
    # st.write('Test set')
    # st.info(X_test.shape)

    # # Loads the Boston House Price Dataset
    # boston = datasets.load_boston()
    # X = pd.DataFrame(boston.data, columns=boston.feature_names)
    # Y = pd.DataFrame(boston.target, columns=["MEDV"])

    # df_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
    # df_boston['target'] = pd.Series(boston.target)
    # st.write(df_boston)

    # # Visualisation
    # chart_select = st.sidebar.selectbox(
    #     label="Type of chart",
    #     options=['Scatterplots', 'Lineplots', 'Histogram', 'Boxplot']
    # )

    # numeric_columns = list(df_boston.select_dtypes(['float', 'int']).columns)

    # if chart_select == 'Scatterplots':
    #     st.sidebar.subheader('Scatterplot Settings')
    #     try:
    #         x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
    #         y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
    #         plot = px.scatter(data_frame=df_boston, x=x_values, y=y_values)
    #         st.write(plot)
    #     except Exception as e:
    #         print(e)
    # if chart_select == 'Histogram':
    #     st.sidebar.subheader('Histogram Settings')
    #     try:
    #         x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
    #         plot = px.histogram(data_frame=df_boston, x=x_values)
    #         st.write(plot)
    #     except Exception as e:
    #         print(e)
    # if chart_select == 'Lineplots':
    #     st.sidebar.subheader('Lineplots Settings')
    #     try:
    #         x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
    #         y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
    #         plot = px.line(df_boston, x=x_values, y=y_values)
    #         st.write(plot)
    #     except Exception as e:
    #         print(e)
    # if chart_select == 'Boxplot':
    #     st.sidebar.subheader('Boxplot Settings')
    #     try:
    #         x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
    #         y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
    #         plot = px.box(df_boston, x=x_values, y=y_values)
    #         st.write(plot)
    #     except Exception as e:
    #         print(e)

    # # Sidebar
    # # Header of Specify Input Parameters
    # st.sidebar.header('Specify Input Parameters')

    # def user_input_features():
    #     CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(
    #         X.CRIM.max()), float(X.CRIM.mean()))
    #     ZN = st.sidebar.slider('ZN', float(X.ZN.min()),
    #                            float(X.ZN.max()), float(X.ZN.mean()))
    #     INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(
    #         X.INDUS.max()), float(X.INDUS.mean()))
    #     CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(
    #         X.CHAS.max()), float(X.CHAS.mean()))
    #     NOX = st.sidebar.slider('NOX', float(X.NOX.min()),
    #                             float(X.NOX.max()), float(X.NOX.mean()))
    #     RM = st.sidebar.slider('RM', float(X.RM.min()),
    #                            float(X.RM.max()), float(X.RM.mean()))
    #     AGE = st.sidebar.slider('AGE', float(X.AGE.min()),
    #                             float(X.AGE.max()), float(X.AGE.mean()))
    #     DIS = st.sidebar.slider('DIS', float(X.DIS.min()),
    #                             float(X.DIS.max()), float(X.DIS.mean()))
    #     RAD = st.sidebar.slider('RAD', float(X.RAD.min()),
    #                             float(X.RAD.max()), float(X.RAD.mean()))
    #     TAX = st.sidebar.slider('TAX', float(X.TAX.min()),
    #                             float(X.TAX.max()), float(X.TAX.mean()))
    #     PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()), float(
    #         X.PTRATIO.max()), float(X.PTRATIO.mean()))
    #     B = st.sidebar.slider('B', float(X.B.min()),
    #                           float(X.B.max()), float(X.B.mean()))
    #     LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(
    #         X.LSTAT.max()), float(X.LSTAT.mean()))
    #     data = {'CRIM': CRIM,
    #             'ZN': ZN,
    #             'INDUS': INDUS,
    #             'CHAS': CHAS,
    #             'NOX': NOX,
    #             'RM': RM,
    #             'AGE': AGE,
    #             'DIS': DIS,
    #             'RAD': RAD,
    #             'TAX': TAX,
    #             'PTRATIO': PTRATIO,
    #             'B': B,
    #             'LSTAT': LSTAT}
    #     features = pd.DataFrame(data, index=[0])
    #     return features

    # df = user_input_features()

    # # Print specified input parameters
    # st.header('Specified Input parameters')
    # st.write(df)
    # st.write('---')

    # # Build Regression Model
    # model = RandomForestRegressor()
    # model.fit(X, Y)
    # # Apply Model to Make Prediction
    # prediction = model.predict(df)

    # st.header('Prediction of Median Value of House (MEDV)')
    # st.write(prediction)
    # st.write('---')

    # # Explaining the model's predictions using SHAP values
    # # https://github.com/slundberg/shap
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X)
    # if st.button('Show SHAP Graphs'):
    #     st.header('Feature Importance')
    #     plt.title('Feature importance based on SHAP values')
    #     shap.summary_plot(shap_values, X)
    #     st.pyplot(bbox_inches='tight')
    #     st.write('---')
    #     plt.title('Feature importance based on SHAP values')
    #     shap.summary_plot(shap_values, X, plot_type="bar")
    #     st.pyplot(bbox_inches='tight')
