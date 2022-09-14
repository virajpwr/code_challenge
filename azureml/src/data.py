import logging
import logzero
import numpy as np
import pandas as pd
import sys
import os
from asyncio.log import logger
from posixpath import split
import warnings
warnings.filterwarnings(action="ignore")


class read_data:
    def __init__(self):
        self.raw_data = '../data/md_raw_dataset.csv'
        self.target_data = '../data/md_target_dataset.csv'

    def data(self) -> pd.DataFrame:
        if os.path.isfile(self.raw_data) and os.path.isfile(self.target_data):
            raw_df = pd.read_csv(self.raw_data, sep=';')
            raw_df = raw_df.rename(columns={'Unnamed: 0': 'index'})
            target_df = pd.read_csv(self.target_data, sep=';')
        else:
            print("File not found")
        return raw_df, target_df


class dataFrame(object):
    def __init__(self, raw_data, target_data):
        self.raw_data = raw_data
        self.target_data = target_data

    def dataframe(self) -> pd.DataFrame:
        final_df = pd.DataFrame()
        for _ in self.raw_data.groups.unique():
            df_train = self.raw_data[self.raw_data.groups == _]
            df_test = self.target_data[self.target_data.groups == _]
            df_merge = pd.merge(df_train, df_test, on='index', how='left')
            final_df = final_df.append(df_merge)
        final_df = final_df.rename(columns={'groups_x': 'groups'})
        final_df = final_df.drop(['groups_y', ], axis=1)
        final_df = final_df.to_parquet('../data/merged_data.parquet')
        return final_df


if __name__ == "__main__":
    read_data = read_data()
    raw_data, target_data = read_data.data()
    dataFrame = dataFrame(raw_data, target_data)
    dataFrame.dataframe()
