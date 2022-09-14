import pandas as pd
import os
from scipy.special import erfinv
import numpy as np
import yaml
import warnings
warnings.filterwarnings(action="ignore")
# Load config file


def load_config(file_path):
    with open(file_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


# calculate time difference between two timestamps.

def cal_time_diff(df, col1, col2, column_name):
    df[col1] = pd.to_datetime(df[col1], errors='coerce')
    df[col2] = pd.to_datetime(df[col2], errors='coerce')
    df[column_name] = df[col1] - df[col2]
    df[column_name] = df[column_name].dt.total_seconds()
    return df

# Split date time column into separate columns into  weekday, day, month


def split_datetime(df, colname):
    df['ts_weekday'] = df[colname].dt.weekday
    df['ts_day'] = df[colname].dt.day
    df['ts_month'] = df[colname].dt.month
    return df

# Handling categorical features.
# To handle categorical features, we will use the following approach:
# 1. Categorify: We will replace the categories with a unique integer id.
# 2. Target encoding: We will replace the categories with the mean of the target variable and smoothed to prevent overfitting.
# 3. Count encode: We will replace the categories with the count of the category in the dataset.


def categorify(df, cat, freq_treshhold=20, unkown_id=1, lowfrequency_id=0):
    '''
    This function is used for label endcoding of categorical features.
    freq_treshhold is used to assign same number to categories with frequency less than freq_treshhold.
    '''
    freq = df[cat].value_counts()
    freq = freq.reset_index()

    freq.columns = [cat, 'count']
    freq = freq.reset_index()

    freq.columns = [cat + '_Categorify', cat, 'count']
    freq[cat + '_Categorify'] = freq[cat + '_Categorify']+2
    freq.loc[freq['count'] < freq_treshhold,
             cat + '_Categorify'] = lowfrequency_id

    freq = freq.drop('count', axis=1)
    df = df.merge(freq, how='left', on=cat)
    df[cat + '_Categorify'] = df[cat + '_Categorify'].fillna(unkown_id)
    return df

# Count encoding done due to high cardinality of categorical columns.


def count_encode(df, col):

    # We keep the original order as cudf merge will not preserve the original order
    df['org_sorting'] = np.arange(len(df), dtype="int32")

    df_tmp = df[col].value_counts().reset_index()
    df_tmp.columns = [col,  'CE_' + col]
    df_tmp = df[[col, 'org_sorting']].merge(
        df_tmp, how='left', left_on=col, right_on=col).sort_values('org_sorting')
    df['CE_' + col] = df_tmp['CE_' + col].fillna(0).values

    df = df.drop('org_sorting', axis=1)
    return df


def target_encode(df, col, target, kfold=5, smooth=20):
    '''
    Taking groupby mean of categorical column and smoothing it using the formula: ((mean_cat*count_cat)+(mean_global*p_smooth)) / (count_cat+p_smooth)
    count_cat := count of the categorical value
    mean_cat := mean target value of the categorical value
    mean_global := mean target value of the whole dataset
    p_smooth := smoothing factor

    To prevent overfitting we use kfold cross validation. 
    To prevent overfitting for low cardinality columns, the means are smoothed with the overall target mean.
    '''
    # We assume that the df dataset is shuffled
    df['kfold'] = ((df.index) % kfold)
    df['org_sorting'] = np.arange(len(df), dtype="int32")
    # We create the output column, we fill with 0
    col_name = '_'.join(col)
    df['TE_' + col_name] = 0.
    for i in range(kfold):
        ###################################
        # filter for out of fold
        # calculate the mean/counts per group category
        # calculate the global mean for the oof
        # calculate the smoothed TE
        # merge it to the original dataframe
        ###################################

        df_tmp = df[df['kfold'] != i]
        mn = df_tmp[target].mean()
        df_tmp = df_tmp[col + [target]
                        ].groupby(col).agg(['mean', 'count']).reset_index()
        df_tmp.columns = col + ['mean', 'count']
        df_tmp['TE_tmp'] = ((df_tmp['mean']*df_tmp['count']) +
                            (mn*smooth)) / (df_tmp['count']+smooth)
        df_tmp_m = df[col + ['kfold', 'org_sorting', 'TE_' + col_name]].merge(
            df_tmp, how='left', left_on=col, right_on=col).sort_values('org_sorting')
        df_tmp_m.loc[df_tmp_m['kfold'] == i, 'TE_' +
                     col_name] = df_tmp_m.loc[df_tmp_m['kfold'] == i, 'TE_tmp']
        df['TE_' + col_name] = df_tmp_m['TE_' + col_name].fillna(mn).values

    ###################################
    # calculate the mean/counts per group for the full dataset
    # calculate the global mean
    # calculate the smoothed TE
    # merge it to the original dataframe
    # drop all temp columns
    ###################################

    df_tmp = df[col + [target]
                ].groupby(col).agg(['mean', 'count']).reset_index()
    mn = df[target].mean()
    df_tmp.columns = col + ['mean', 'count']
    df_tmp['TE_tmp'] = ((df_tmp['mean']*df_tmp['count']) +
                        (mn*smooth)) / (df_tmp['count']+smooth)

    df = df.drop('kfold', axis=1)
    df = df.drop('org_sorting', axis=1)
    return df

# Perform Gaussian rank normalization on continuous features and target variable.
# This is done to reduce the effect of outliers. since the boosting is sensitive to outliers.

def gaussrank_gpu(data, epsilon=1e-6):
    r_gpu = data.argsort().argsort()  # compute rank
    r_gpu = (r_gpu/r_gpu.max()-0.5)*2  # scale to (-1,1)
    r_gpu = np.clip(r_gpu, -1+epsilon, 1-epsilon)  # clip the values to epsilon
    # apply inverse error function to get normal distribution.
    r_gpu = erfinv(r_gpu)
    return (r_gpu)
