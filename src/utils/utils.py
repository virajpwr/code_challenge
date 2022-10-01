from imports import *


def find_remove_duplicates(list_of_values: list) -> list:
    """
    __summary__: This function is used to remove duplicates from a list to return unique values.
    parameters:
        list_of_values {list} -- [list of values]
    returns:
        output {list} -- [list of unique values]
    """
    output = []
    seen = set()
    for value in list_of_values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


def return_dictionary_list(lst_of_tuples: list) -> list:
    """ 
    __summary__: This function is used to convert a list of tuples to a list of dictionaries.
    parameters:
        lst_of_tuples {list} -- [list of tuples]    
    returns:
        lst_of_dicts {list} -- [list of dictionaries]
   """
    orDict = defaultdict(list)
    # iterating over list of tuples
    for key, val in lst_of_tuples:
        orDict[key].append(val)
    return orDict


def logs(path: str, file: str) -> object:
    """[Create a log file to record the experiment's logs]
    parameters:
        path {string} -- path to the directory
        file {string} -- file name
    Returns:
        [object] -- [logger that record logs]
    """

    # check if the file exists
    log_file = os.path.join(path, file)

    if not os.path.exists(log_file):
        open(log_file, "w+").close()

    logging_format: str = "%(levelname)s: %(asctime)s: %(message)s"

    # configure the logger
    logging.basicConfig(level=logging.INFO, format=logging_format)
    logger = logging.getLogger()

    # create a file handler for output file
    file_handler = logging.FileHandler(log_file)

    # set the logging level for log file
    file_handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter(logging_format)

    # set the format for file handler
    file_handler.setFormatter(formatter)

    if (logger.hasHandlers()):
        logger.handlers.clear()

    # add the handlers to the logger

    logger.addHandler(file_handler)

    return logger


def left_subtract(l1: list, l2: list) -> list:
    # Used for removing the common elements from two lists
    """
    l1: list
    l2: list
    return: list
    """
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst


def load_config(file_path: str) -> dict:
    """
    __summary__: This function is used to load the config file.
    parameters:
        file_path {str} -- [path to the config file]
    returns:
        config {dict} -- [config file]
    """
    with open(file_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def cal_time_diff(df: pd.DataFrame, col1: pd.Series, col2: pd.Series, column_name: str) -> pd.DataFrame:
    """
    __summary__: This function is used to calculate the time difference between two timestamps.

    parameters:
        df {pd.DataFrame} -- [dataframe]
        col1 {pd.Series} -- [first timestamp column]
        col2 {pd.Series} -- [second timestamp column]
        column_name {str} -- [name of the new column]
    returns:
        df {pd.DataFrame} -- [dataframe with new column]
    """
    df[col1] = pd.to_datetime(df[col1], errors='coerce')
    df[col2] = pd.to_datetime(df[col2], errors='coerce')
    df[column_name] = df[col1] - df[col2]
    df[column_name] = df[column_name].dt.total_seconds()
    return df


def anova(data: pd.DataFrame, i: str) -> float:
    """"
    __summary__: This function is used to calculate the anova score for each feature.
    parameters:
        data {pd.DataFrame} -- [dataframe]  
        i {str} -- [name of the feature]
    returns:
        p-value {float} -- p-value of the anova test
    """
    logger = logs(path="logs/", file="utils.logs")
    logger.info("Performing anova test for feature: {}".format(i))
    try:
        from scipy import stats
        grps = pd.unique(data[i].values)
        d_dat = {grp: data[i][data[i] == grp] for grp in grps}
        k = len(pd.unique(data[i]))  # number of conditions
        N = len(data.values)  # conditions times participants
        n = data.groupby(i).size()[0]  # Participants in each condition
        ss_between = (sum(data.groupby(i).sum()[
                      'target'] ** 2)/n) - (data['target'].sum() ** 2)/N  # calculate sum of squares between groups
        # calculate sum of squares within groups
        sum_y_pred = sum([value**2 for value in data['target'].values])
        # calculate sum of squares total
        ss_within = sum_y_pred - sum(data.groupby(i).sum()['target'] ** 2)/n
        ss_total = sum_y_pred - (data['target'].sum() ** 2)/N
        ms_between = ss_between/(k-1)  # calculate mean square between groups
        ms_within = ss_within/(N-k)  # calculate mean square within groups
        f = ms_between/ms_within  # calculate f-statistic
        p = stats.f.sf(f, k-1, N-k)  # calculate p-value
        eta_squared = ss_between/ss_total
        omega_squared = (ss_between - (k-1)*ms_within)/(ss_total + ms_within)
        # print(f, p)
        return p
    except Exception as e:
        logger.error(
            "Error in performing anova test for feature: {}".format(i))
        logger.error('error: {}'.format(e))
        print(e)


def split_datetime(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    """_summary_: This function is used to split the datetime column into separate columns.

    parameters:
        df {pd.DataFrame} -- [dataframe]
        colname {str} -- [name of the datetime column]
    returns:
        df {pd.DataFrame} -- [dataframe with new columns]
    """
    df['ts_weekday'] = df[colname].dt.weekday
    df['ts_day'] = df[colname].dt.day
    df['ts_month'] = df[colname].dt.month
    weekday_one_hot = pd.get_dummies(df['ts_weekday'], prefix='ts_weekday')
    df = pd.concat([df, weekday_one_hot], axis=1)
    return df


# Handling categorical features.
# To handle categorical features, we will use the following approach:
# 1. Categorify: We will replace the categories with a unique integer id.
# 2. Target encoding: We will replace the categories with the mean of the target variable and smoothed to prevent overfitting.
# 3. Count encode: We will replace the categories with the count of the category in the dataset.


def categorify(df: pd.DataFrame, cat: str, freq_treshhold=20, unkown_id=1, lowfrequency_id=0) -> pd.DataFrame:
    """__summary__: This function is used perform label encoding on the categorical features. 
    A frequency threshold is used to replace the categories with low frequency with a single category. 
    To deal with high cardinality and overfitting, we will replace the categories with low frequency with a single category.
    The category id 1 or 0 is reserved for unknown and low frequency categories respectively. 
    parameters: 
        df {pd.DataFrame} -- [dataframe]
        cat {str} -- [name of the categorical column]
        freq_treshhold {int} -- [frequency threshold]
        unkown_id {int} -- [to fil nan values]
        lowfrequency_id {int} -- [to replace low frequency categories]

    """
    freq = df[cat].value_counts()  # frequency of each category
    freq = freq.reset_index()  # reset the index

    freq.columns = [cat, 'count']  # rename the columns
    freq = freq.reset_index()  # reset the index

    freq.columns = [cat + '_Categorify', cat, 'count']  # rename the columns
    freq[cat + '_Categorify'] = freq[cat +
                                     '_Categorify']+2  # add 2 to the index
    freq.loc[freq['count'] < freq_treshhold,
             cat + '_Categorify'] = lowfrequency_id  # replace low frequency categories with 0

    freq = freq.drop('count', axis=1)  # drop the count column
    # merge the frequency dataframe with the original dataframe
    df = df.merge(freq, how='left', on=cat)
    # fill nan values with 1
    df[cat + '_Categorify'] = df[cat + '_Categorify'].fillna(unkown_id)
    # convert the column to category type
    df[cat + '_Categorify'] = df[cat + '_Categorify'].astype('category')
    return df


def count_encode(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """"
    __summary__: This function is used to perform count encoding on the categorical features. Count encoding done due to high cardinality of categorical columns.
                It calculates the frequency from one or more categorical features.
    Count Encoding (CE) calculates the frequency from one or more categorical features given the training dataset.

    Count Encoding creates a new feature, which can be used by the model for training.
     It groups categorical values based on the frequency together.
     Count Encoding creates a new feature, which can be used by the model for training. It groups categorical values based on the frequency together.
    parameters:
        df {pd.DataFrame} -- [dataframe]
        col {str} -- [name of the categorical column]
    returns:
        df {pd.DataFrame} -- [dataframe with new column]
    """
    # We keep the original order as cudf merge will not preserve the original order
    df['org_sorting'] = np.arange(len(df), dtype="int32")

    # count the number of each category
    df_tmp = df[col].value_counts().reset_index()
    df_tmp.columns = [col,  'CE_' + col]  # rename the columns
    df_tmp = df[[col, 'org_sorting']].merge(
        df_tmp, how='left', left_on=col, right_on=col).sort_values('org_sorting')  # merge the count with the original dataframe
    # fill the missing values with 0
    df['CE_' + col] = df_tmp['CE_' + col].fillna(0).values
    df = df.drop('org_sorting', axis=1)  # drop the temporary column
    # convert the column to int32
    df['CE_' + col] = df['CE_' + col].astype('int32')
    return df


def target_encode(df: pd.DataFrame, col: str, target: str, kfold=5, smooth=20) -> pd.DataFrame:
    '''
    __summary__: This function is used to perform target encoding on the categorical features. Target encoding done due to high cardinality of categorical columns.
    Taking groupby mean of categorical column and smoothing it using the formula: 

    TE = ((mean_cat*count_cat)+(mean_global*p_smooth)) / (count_cat+p_smooth)

    count_cat := count of the categorical value
    mean_cat := mean target value of the categorical value
    mean_global := mean target value of the whole dataset
    p_smooth := smoothing factor

    In smoothing 
    1. if the number of observation is high, we want to use the mean of this category value
    2. if the number of observation is low, we want to use the global mean

    To prevent overfitting we use kfold cross validation. 
    To prevent overfitting for low cardinality columns, the means are smoothed with the overall target mean.

    parameters:
        df {pd.DataFrame} -- [dataframe]
        col {str} -- [name of the categorical column]
        target {str} -- [name of the target column]
        kfold {int} -- [number of folds]
        smooth {int} -- [smoothing factor]
    returns:
        df {pd.DataFrame} -- [dataframe with new column]

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


def gaussrank_gpu(data: np.array, epsilon=1e-6) -> np.array:
    """
        __summary__: This function performs gaussian rank normalization on continous features and target variable. 
        It the data from arbitrary distribution to Gaussian distribution based on ranks. This is done to reduce the effect of outliers.
        parameter:
            data: np.array
            epsilon: float
        return:
            np.array
    """
    r_gpu = data.argsort().argsort()  # compute rank
    r_gpu = (r_gpu/r_gpu.max()-0.5)*2  # scale to (-1,1)
    r_gpu = np.clip(r_gpu, -1+epsilon, 1-epsilon)  # clip the values to epsilon
    # apply inverse error function to get normal distribution.
    r_gpu = erfinv(r_gpu)
    return (r_gpu)


def merge_data_func(raw_data: pd.DataFrame, target_data: pd.DataFrame) -> pd.DataFrame:
    """
    [This function will create a dataframe from the raw data and target data]

    Arguments:
        raw_data {pd.DataFrame} -- [raw data]
        target_data {pd.DataFrame} -- [target data]

    Returns:
        final_df{pd.DataFrame} -- [dataframe]
    """

    final_df = pd.DataFrame()  # Create an empty dataframe

    # Loop through unqiue values of group and join the groups number on id and append to final_df (dataframe).

    for _ in raw_data.groups.unique():
        df_raw = raw_data[raw_data.groups == _]
        df_target = target_data[target_data.groups == _]
        df_merge = pd.merge(df_target, df_raw, on='index')
        final_df = pd.concat([final_df, df_merge], axis=0)
    return final_df


def interpolate_date_time_features(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    [This function will interpolate the date time features]

    Arguments:
        df {pd.DataFrame} -- [dataframe]

    Returns:
        df{pd.DataFrame} -- [dataframe]
    """
    df = shuffle(df)
    # Interpolate the date time features
    for col in df[config]:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col] = df[col].values.astype('int64')
        df[col][df[col] < 0] = np.nan
        df[col] = pd.to_datetime(df[col].interpolate(), unit='ns')
    return df


def calculate_time_difference(df: pd.DataFrame,
                              col1: pd.Series, col2: pd.Series) -> str:
    """[Calculate the time difference between two timestamps]

    Arguments:
        start_time {float} -- [start time]
        end_time {float} -- [end time]

    Returns:
        [str] -- [time difference]
    """
    time_difference: pd.timedelta[ns] = end_time - start_time
    return str(datetime.timedelta(seconds=time_difference))


def remove_outliers(df: pd.DataFrame, cols: pd.Series) -> pd.DataFrame:
    """[Remove outliers from the dataframe]

    Arguments:
        df {pd.DataFrame} -- [dataframe]
        cols {pd.Series} -- [columns]

    Returns:
        [pd.DataFrame] -- [dataframe]
    """
    q1 = df[cols].quantile(0.25)
    q3 = df[cols].quantile(0.75)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - (1.5 * iqr)
    fence_high = q3 + (1.5 * iqr)
    df = df.loc[(df[cols] > fence_low) & (df[cols] < fence_high)]
    return df
