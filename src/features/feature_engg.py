
from imports import *


class FeatEngg(object):
    """_summary_: This function is used to create new features from existing features.

    Parameters:
        df {dataframe}: A dataframe with the raw data
        config {dict}: A dictionary with the configuration parameters
    methods:
        categorify_columns(): A function to label encode the categorical columns

        target_encode_columns(): A function to take groupy mean of the target column for categorical columns.

        count_encode_columns(): A function to take count (value_count) of the categorical columns.

        transforming_target_continuous(): A function to transform the target and continous columns distribution to guassian distribution using gaussrank. 

        split_datetime_col(): A function to split the datetime columns into year, month, day for the column 'when'.

        cal_time_diff(): A function to calculate the time difference between two datetime columns.
    """

    def __init__(self, df, config, logger: Logger) -> None:
        """
        __Summary__: Initialize the class

        Parameters:
            df {dataframe}: A dataframe with the raw data
            config {dict}: A dictionary with the configuration parameters
        returns:
            None
        """
        self.df = df
        self.config = config
        self.logger = logger

    def categorify_columns(self):
        """_summary_: A function to label encode the categorical columns using categorify function from utils.
        parameters:
            None
        returns:
            df {dataframe}: A dataframe with the label encoded columns

        """
        self.logger.info("Label encoding the categorical columns")

        for col in self.config["categorify_columns"]:
            self.df = categorify(df=self.df, cat=col, freq_treshhold=20)
            self.logger.info(
                "performing label encoding on column: {}".format(col))
        return self.df

    def target_encode_columns(self):
        self.logger.info("Target encoding the categorical columns")
        """"
        __Summary__: A function to take groupy mean of the target column for categorical columns using target_encode function from utils.
        parameters:
            None
        returns:
            df {dataframe}: A dataframe with the target encoded columns
        """
        for col in self.config["target_encode_columns"]:
            self.df = target_encode(self.df, [col], 'target')
            self.logger.info(
                "performing target encoding on column: {}".format(col))
        return self.df

    def count_encode_columns(self):
        self.logger.info("Count encoding the categorical columns")
        """_summary_: A function to take count (value_count) of the categorical columns using count_encode function from utils.
        parameters:
            None
        returns:
            df {dataframe}: A dataframe with the count encoded columns

        for col in self.config["count_encode_columns"]:
            self.df = count_encode(self.df, col)
        return self.df
        """
        for col in self.config["count_encode_columns"]:
            self.df = count_encode(self.df, col)
            self.logger.info(
                "performing count encoding on column: {}".format(col))
        return self.df

    def transforming_target_continuous(self):
        """_summary_: A function to transform the target and continous columns distribution to guassian distribution using
                     gaussrank_gpu function from utils.
        parameters:
            None
        returns:
            df {dataframe}: A dataframe with the transformed columns
        """
        self.logger.info(
            "Transforming the target and continous columns distribution to guassian distribution")
        self.df["target"] = gaussrank_gpu(self.df["target"])
        for col in self.config["continous_cols"]:
            self.df[col] = gaussrank_gpu(self.df[col].values)
            self.logger.info("performing gaussrank on column: {}".format(col))
        return self.df

    def standard_scalar(self):
        """_summary_: A function to standardize the continous columns using standard_scalar function from utils.
        parameters:
            None
        returns:
            df {dataframe}: A dataframe with the standardized columns
        """
        self.logger.info("Standardizing the continous columns")
        for col in self.config["continous_cols"]:
            self.df = standard_scalar(self.df, col)
            self.logger.info(
                "performing standard scaling on column: {}".format(col))

        return self.df
        
    def split_datetime_col(self):
        """_summary_: A function to split the datetime columns into year, month, day for the column 'when'.
        parameters:
            None
        returns:
            df {dataframe}: A dataframe with the split datetime columns
        """
        self.logger.info(
            "Splitting the datetime column into year, month, day, weekday")
        # Split datetime columns into year, month, day.
        for colname in self.df[self.config['date_cols']]:
            self.df[colname] = self.df[colname].astype('datetime64[ns]')
        # Using split_datetime function from utils.
        self.df = split_datetime(self.df, "when")
        return self.df

    def cal_time_diff(self):
        """_summary_: A function to calculate the time difference between two datetime columns.
        parameters:
            None
        returns:
            df {dataframe}: A dataframe with the time difference columns

        """
        self.logger.info(
            "Calculating the time difference between two datetime columns")
        # Calculate time difference between two datetime columns.
        self.df = cal_time_diff(self.df, 'process_end',
                                'start_process', 'time_diff_process')
        self.df = cal_time_diff(
            self.df, 'subprocess1_end', 'start_subprocess1', 'time_diff_subprocess')
        return self.df
