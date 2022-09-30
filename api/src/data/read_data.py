from logging import Logger
from imports import *


class read_data(object):
    """_summary_: A class to read the raw data and target data from the data/raw folder

    Methods:
        read_data(): A function to check if data exists and read the raw data and target data from the data/raw folder
    """

    def __init__(self, logger: Logger) -> None:
        """_summary_: initialize the class
        Returns:
            None
        """
        self.raw_data = './data/raw/md_raw_dataset.csv'
        self.target_data = './data/raw/md_target_dataset.csv'
        self.logger = logger

    def read_data(self) -> pd.DataFrame:
        """__summary__: A function to read the raw data and target data csv from the data/raw folder
            parameters:
                None
            Returns:
                raw_df {dataframe}: A dataframe containing the raw data
                target_df {dataframe}: A dataframe containing the target data
        """
        # set a logger file
        # logger = logs(path="logs/", file="data.logs")

        self.logger.info(
            "Reading the raw data and target data from the data/raw folder")

        # Check if file exists and read the data
        if os.path.isfile(self.raw_data) and os.path.isfile(self.target_data):
            raw_df = pd.read_csv(self.raw_data, ';')
            target_df = pd.read_csv(self.target_data, ';')
            self.logger.info(
                "Successfully read the raw data and target data from the data/raw folder")
            return raw_df, target_df
        else:
            self.logger.error("The raw data and target data are not available")


class merge_dataset(object):
    """_summary_: A class to merge the raw data and target data and save the merged data to the data/raw folder
    attributes:
        raw_data {dataframe}: A dataframe containing the raw data
        target_data {dataframe}: A dataframe containing the target data
    Methods:
        merge_dataset(): A function to merge the raw data and target data on index and groups.

    """

    def __init__(self, raw_data, target_data, logger: Logger) -> None:
        """_summary_: initialize the class
        Attributes:
            raw_data {dataframe}: A dataframe containing the raw data
            target_data {dataframe}: A dataframe containing the target data
        Returns:
            None

        """

        self.raw_data = raw_data
        self.target_data = target_data
        self.logger = logger

    def merge_dataset(self):
        """_summary_: A function to merge the raw data and target data

        Returns:
            merged_df {dataframe}: A dataframe containing the merged raw data and target data
        """
        # set a logger file
        # logger = logs(path="logs/", file="data.logs")

        self.logger.info("Merging the raw data and target data")
        # Lowercase the column names
        self.raw_data.columns = self.raw_data.columns.str.lower()
        self.target_data.columns = self.target_data.columns.str.lower()
        # rename unnamed columns. unnamed column is the index of the raw data.
        self.raw_data = self.raw_data.rename(columns={
                                             'unnamed: 0': 'index', 'unnamed: 17': 'unnamed_17', 'unnamed: 7': 'unnamed_7'})

        # Fill missing value of groups with the mode of the groups on date 21/01/2020.

        self.raw_data['groups'] = self.raw_data['groups'].fillna(value=int(
            self.raw_data['groups'].loc[self.raw_data['when'] == '21/12/2020'].mode()))

        # converting the index and groups column to int type
        self.raw_data[['index', 'groups']] = self.raw_data[['index', 'groups']].astype(
            int)

        # converting the index and groups column to int type
        self.target_data[['index', 'groups']] = self.target_data[[
            'index', 'groups']].astype(int)

        # use merge function from utils to join raw data and target data on index and groups
        merged_df = merge_data_func(self.raw_data, self.target_data)
        merged_df = merged_df.rename(columns={'groups_x': 'groups'})
        merged_df = merged_df.drop(['index', 'groups_y'], axis=1)
        # save the merged data to the data/raw folder as parquet file
        merged_df.to_parquet('./data/raw/merged_data.parquet')
        self.logger.info("Merge successful")
        return merged_df
