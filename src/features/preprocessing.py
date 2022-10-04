
from imports import *


class preprocessing(object):
    """_summary_: A class to preprocess the raw data and target data

    parameters:
        df {dataframe}: A dataframe with the raw data
        config {dict}: A dictionary with the configuration

    Methods:
        convert_dtypes(): A function to convert the data types of the dataframe

        replace_values(): A function to replace the values of the columns [cycle, crystal_supergroup, etherium_before_start]

        interpolate_datetime(): A function to interpolate the datetime columns

        drop_duplicate_rows(): A function to drop the duplicate rows

        processing_missing_values(): A function to fill na with median for continous columns and mode for categorical columns

        remove_outliers(): A function to remove the outliers
    """

    def __init__(self, df: pd.DataFrame, config: dict, logger: Logger) -> None:
        self.df = df
        self.config = config
        self.logger = logger

    def convert_dtypes(self) -> pd.DataFrame:
        """_summary_: A function to convert the data types of the dataframe
        parameters:
            None
        Returns:
            df {dataframe}: A dataframe with the converted data types
        """

        self.logger.info("Converting dtypes")

        # self.df[self.config['cat_cols']] = self.df[self.config['cat_cols']].astype(
        #     'category', errors='ignore')  # converting the categorical columns to category type
        self.df[self.config['cat_cols']] = self.df[self.config['cat_cols']].astype(
            'int64', errors='ignore')  # converting the categorical columns to int type
        self.df[self.config['continous_cols_before_preprocess']] = self.df[self.config['continous_cols_before_preprocess']].astype(
            'float', errors='ignore')  # converting the continous columns to float type

        return self.df

    def replace_values(self) -> pd.DataFrame:
        """_summary_: A function to replace the values of the columns [cycle, crystal_supergroup, etherium_before_start]
        parameters:
            None
        Returns:
            df {dataframe}: A dataframe with the replaced values
        """

        self.logger.info("Replacing values")

        self.df['cycle'] = self.df['cycle'].replace(
            ['33', '1ª', '2ª', '3ª', '131'], ['1', '1', '2', '3', '4'])  # replace the categorical values.
        self.df['crystal_supergroup'] = self.df['crystal_supergroup'].replace(
            '1ª', '0')  # replace the unknown value by 0.
        self.df['etherium_before_start'] = self.df['etherium_before_start'].replace(
            ['21/12/2020 12:11'], 441.78)  # replace the date value with the mean of the column.
        return self.df

    def interpolate_datetime(self) -> pd.DataFrame:
        """_summary_: A function to interpolate the datetime columns
        parameters:
            None
        Returns:
            df {dataframe}: A dataframe with the interpolated datetime column
        """

        self.logger.info("Interpolating dattime columns")

        self.df = shuffle(self.df)  # suffling the dataframe
        self.df = interpolate_date_time_features(
            self.df, self.config['date_cols'])  # interpolating the datetime columns
        self.df['start_critical_subprocess1'] = pd.to_datetime(
            self.df['start_critical_subprocess1'], errors='coerce')  # converting the datetime columns to datetime type
        self.df['start_critical_subprocess1'] = self.df['start_critical_subprocess1'].values.astype(
            'int64')  # converting the datetime columns to int type
        self.df['start_critical_subprocess1'][self.df['start_critical_subprocess1']
                                              < 0] = np.nan  # replacing the negative values with nan
        self.df['start_critical_subprocess1'] = pd.to_datetime(
            self.df['start_critical_subprocess1'].interpolate(), unit='ns')  # interpolating the datetime columns
        self.df[self.config['date_cols']
                ] = self.df[self.config['date_cols']].astype('datetime64[ns]')  # converting the datetime columns to datetime type
        self.df['start_critical_subprocess1'] = self.df['start_critical_subprocess1'].astype(
            'datetime64[ns]')  # converting the datetime columns to datetime type
        return self.df  # returning the dataframe with interpolated datetime columns

    def drop_duplicate_rows(self) -> pd.DataFrame:
        """_summary_: A function to drop the duplicates
        parameters:
            None
        Returns:
            df {dataframe}: A dataframe with the dropped duplicates
        """

        self.logger.info(
            "Dropping duplicates rows for columns: {}".format('start_process'))

        self.df['start_process'] = self.df['start_process'].astype(
            'datetime64[ns]')  # converting the datetime columns to datetime type
        # sorting the dataframe by start_process to remove the duplicates
        self.df = self.df.sort_values(by='start_process')
        # Drop duplicate and keep the first row
        self.df = self.df.drop_duplicates('start_process', keep='first')
        return self.df

    # def processing_missing_values(self) -> pd.DataFrame:
    #     """_summary_: A function to process the missing values using MICE
    #     parameters:
    #         None
    #     Returns:
    #         df {dataframe}: A dataframe with the processed missing values
    #     """
    #     cols_imputed = mice(self.df[self.config['vars_with_na']].values)
    #     imputed_df = pd.DataFrame(
    #         cols_imputed, columns=self.df[self.config['vars_with_na']].columns)
    #     df_ = self.df.drop(self.df[self.config['vars_with_na']], axis=1)
    #     df_ = df_.reset_index()
    #     imputed_df_ = imputed_df.reset_index()
    #     df_imputed_final = pd.merge(df_, imputed_df_, on='index')
    #     return df_imputed_final

    def processing_missing_values(self) -> pd.DataFrame:
        """_summary_: A function to fill na with median for continous columns and mode for categorical columns
        parameters:
            None
        Returns:
            df {dataframe}: A dataframe with the processed missing values
        """
        self.logger.info("Processing missing values")
        for col in self.config["vars_with_na"]:
            self.df["NA_" + col] = self.df[col].isna().astype(np.int8)
            if str(self.df[col].dtypes) in ['int64', 'float64']:
                self.df[col].fillna(self.df[col].median(), inplace=True)
            else:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        return self.df

    def drop_duplicate_columns(self) -> pd.DataFrame:
        """_summary_: A function to drop the duplicate columns
        parameters:
            None
        Returns:
            df {dataframe}: A dataframe with the dropped duplicate columns
        """
        self.logger.info("Dropping duplicate columns")
        self.df = self.df.drop(
            columns=self.config['duplicate_cols'])  # dropping the duplicate columns unnamed_17
        # self.df = self.df.T.drop_duplicates().T # Doesnt work
        return self.df

    def remove_outliers(self) -> pd.DataFrame:
        """_summary_: A function to remove the outliers
        parameters:
            None
        Returns:
            df {dataframe}: A dataframe with the removed outliers
        """
        self.logger.info("Removing outliers IQR method")
        for i in self.df[self.config['continous_cols_before_preprocess']]:
            # removing the outliers using the remove_outliers function from utils.
            df_temp = remove_outliers(self.df, i)
            self.logger.info("outlier removed for column: {}".format(i))
        return df_temp
