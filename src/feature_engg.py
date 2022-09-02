import logging
import numpy as np
from src.utils import gaussrank_gpu
from src.utils import split_datetime, load_config, categorify, target_encode, count_encode
import logzero
import pandas as pd
import sys
import os
from asyncio.log import logger
from lib2to3.pgen2.pgen import DFAState
from posixpath import split
import warnings
warnings.filterwarnings(action="ignore")


class FeatEngg(object):
    def __init__(self, df, config) -> None:
        self.df = df
        self.config = config

    def feat_engg_date(self):
        self.df["expected_total_time_to_report"] = (
            self.df['reported_on_tower']) - (self.df['expected_start'])
        self.df["expected_total_time_to_report"] = (
            self.df['reported_on_tower']) - (self.df['start_process'])
        self.df["expected_total_time_to_report"] = (
            self.df['predicted_process_end']) - (self.df['start_process'])
        self.df["expected_total_time_to_report"] = (
            self.df['subprocess1_end']) - (self.df['start_process'])

    def categorify_columns(self):
        for col in self.config["categorify_columns"]:
            self.df = categorify(df=self.df, cat=col, freq_treshhold=20)
        return self.df

    def target_encode_columns(self):
        for col in self.config["target_encode_columns"]:
            self.df = target_encode(self.df, [col], 'target')
        return self.df

    def count_encode_columns(self):
        for col in self.config["count_encode_columns"]:
            self.df = count_encode(self.df, col)
        return self.df

    def transforming_target_continuous(self):
        self.df["target"] = gaussrank_gpu(self.df["target"])
        for col in self.config["continuous_columns"]:
            self.df[col] = gaussrank_gpu(self.df[col].values)
        return self.df
