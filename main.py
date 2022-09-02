from src.train import model_training
from src.feature_selection import feature_selection, save_features_after_feature_selection
from src.feature_engg import FeatEngg
from src.preprocessing import Preprocessing
from src.utils import load_config
from src.data import dataFrame
import pandas as pd
import os
import warnings
warnings.filterwarnings(action="ignore")


def main():

    config = load_config("config.yml")  # Load config file
    df = pd.read_parquet("./data/merged_data.parquet")  # Load merged data

    # 1. Preprocessing
    preprocessing = Preprocessing(df, config)
    preprocessing.processing_missing_values()
    df = preprocessing.preprocess_date_cols(config["date_cols"])

    # 2. Feature engineering
    feat_engg = FeatEngg(df=df, config=config)
    # df = feat_engg.feat_engg_date()
    df = feat_engg.categorify_columns()
    df = feat_engg.target_encode_columns()
    df.head()
    df = feat_engg.count_encode_columns()
    df = feat_engg.transforming_target_continuous()

    # 3. feature selection
    selected_features = feature_selection(
        df, config["required_columns"], df["target"])
    save_features_after_feature_selection(selected_features, config)
    cols_to_drop = ["Cycle", "super_hero_group",
                    "opened", "crystal_supergroup", "groups", "index"]
    selected_features = [e for e in selected_features if e not in cols_to_drop]
    print(selected_features)
    target = df['target']
    train_model = model_training(df, target, selected_features, config)
    train_model.convert_to_DMatrix()
    train_model.train_baseline_model()
    error = train_model.evaluate_model()  # evaluate the model
    train_model.save_model()

    # train_model.hyperparameter_search() # Use this line to find the best hyperparameters. Not utilised in the final model.


if __name__ == "__main__":
    main()
