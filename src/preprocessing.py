import logging
from src.utils import split_datetime, load_config, cal_time_diff
import logzero
import numpy as np
import pandas as pd
import sys
import os
from asyncio.log import logger
from posixpath import split
import warnings
warnings.filterwarnings(action="ignore")


class Preprocessing(object):
    def __init__(self, df, config) -> None:
        self.df = df
        self.config = config

    def processing_missing_values(self):
        # Fll missing values of continous variables with median and categorical variables with mode.
        # vars_with_na = [var for var in self.df.columns if self.df[var].isnull().sum() > 0]
        for col in self.config["vars_with_na"]:
            self.df["NA_" + col] = self.df[col].isna().astype(np.int8)
            if str(self.df[col].dtypes) in ['int64', 'float64']:
                self.df[col].fillna(self.df[col].median(), inplace=True)
            else:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)

    def preprocess_date_cols(self, col_names):
        # Split datetime columns into year, month, day.
        for colname in self.df[col_names]:
            self.df[colname] = pd.to_datetime(self.df[colname])
        self.df = split_datetime(self.df, "when")
        return self.df

    def cal_time_diff(self, col_names):
        # Calculate time difference between two datetime columns.
        self.df = cal_time_diff(self.df, 'process_end',
                                'start_process', 'time_diff_process')
        self.df = cal_time_diff(
            self.df, 'subprocess1_end', 'start_subprocess1', 'time_diff_subprocess')
        return self.df
