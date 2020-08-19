"""Data Cleaning

This script allows the user to process data cleaning for both CoachingMate data and Garmin data.

This script requires that `pandas`, `numpy` be installed within the Python
environment you are running this script in.

This file can also be imported as a module
"""

import numpy as np
import pandas as pd
import os
import utility
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from data_loader import DataLoader
from configparser import ConfigParser

# Set the data frame display option
pd.set_option('display.max_row', 200)
pd.set_option('display.max_columns', 20)

class OriginalDataCleaner():
    """
    A class used to process data cleaning on CoachingMate data

    ...

    Attributes
    ----------
    file_name : str
       The name of the CoachingMate data file in csv format

    Methods
    -------
    replace_missing_vals_with_nan()
       Replace different kinds of missing values with Nah
    formalize_dates()
        Formalize the dates to YYYY-mm-dd HH:MM:SS form
    process_data_cleaning()
        Process the data cleaning
    """

    def __init__(self):
        self.numerical_columns = utility.get_original_numerical()
        self.categorical_columns = utility.get_original_categorical()

    def show_missing_val_distribution(self):
        pass

    def show_missing_val_locations(self):
        pass

    def make_heatmap(self):
        pass

    def make_correlation_plot(self):
        pass

    def replace_missing_vals_with_nan(self):
        pass

    def formalize_dates(self):
        pass

    def clean_numerical_columns(self, dataframe: pd.DataFrame, columns_focus_on=None):
        pass

    def clean_categorical_columns(self, dataframe: pd.DataFrame, columns_focus_on=None):
        pass

    def process_data_cleaning(self, dataframe: pd.DataFrame):
        """
        Returns
        -------
        cleaned_df : pandas DataFrame
            Cleaned athlete CoachingMate data
        """
        self.clean_numerical_columns(dataframe)
        self.clean_categorical_columns(dataframe)
        return dataframe


class AdditionalDataCleaner():
    """
    A class used to process data cleaning on a group of Garmin datasets for one athlete

    ...

    Attributes
    ----------
    file_names : list(str)
      The list of names of the additional data for a certain athlete

    Methods
    -------
    process_data_cleaning()
       Process the data cleaning
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.numerical_ordered_columns = utility.get_additional_numerical_ordered()
        self.numerical_fluctuating_columns = utility.get_additional_numerical_fluctuating()
        self.categorical_columns = utility.get_additional_categorical()
        self.dataframe = dataframe

    def check_empty(self):
        """Check whether the data frame is empty

        Returns
        -------
        boolean
           Whether the data frame is empty
        """
        return self.dataframe.empty

    def format_missing_val_with_nan(self):
        self.dataframe.replace('None', np.nan)

    def check_missing_val_perc(self):
        # if columns_focus_on:
        #     missing_val_perc = dataframe[columns_focus_on].isnull().sum() / dataframe.shape[0]
        # else:
        #     missing_val_perc = dataframe.isnull().sum() / dataframe.shape[0]
        missing_val_perc = self.dataframe.isnull().sum() / self.dataframe.shape[0]
        return missing_val_perc

    def convert_str_to_num_in_numerical_cols(self):
        for column in self.numerical_ordered_columns+self.numerical_fluctuating_columns:
            # dataframe[column] = dataframe[column].astype(float)
            self.dataframe[column] = pd.to_numeric(self.dataframe[column], errors='coerce')

    def check_outliers(self):
        z = np.abs(stats.zscore(self.dataframe[self.numerical_ordered_columns]))
        threshold = 3
        outlier_zscores = np.where(z > threshold)
        outlier_rows_cols = outlier_zscores
        if len(outlier_zscores[0]) > 0:
            for row in outlier_zscores[0]:
                for col in outlier_zscores[1]:
                    outlier_zscore = z[row][col]
                    ourlier_value = self.dataframe.iloc[row, col]
        return outlier_rows_cols

    def drop_column(self, column):
        pass

    def add_column_time_in_seconds(self):
        time_in_seconds = []
        self.dataframe.insert(3, 'Time in Seconds', time_in_seconds, True)
        pass

    def apply_univariate_imputation(self, column):
        imp = SimpleImputer(strategy="most_frequent")

        trainingData = self.dataframe.iloc[:, :].values
        dataset = self.dataframe.iloc[:, :].values

        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        imputer = imputer.fit(trainingData[[column]])
        dataset[[column]] = imputer.transform(dataset[[column]])

        # self.dataframe[column] = imp.fit_transform(self.dataframe[column])

    def apply_multivariate_imputation(self):
        pass

    def apply_nearest_neighbor_imputation(self, missing_values=np.nan, strategy='mean'):
        pass

    def handle_missing_values(self):
        self.format_missing_val_with_nan()
        missingl_perc = self.check_missing_val_perc()
        print(missingl_perc)
        for column in self.numerical_ordered_columns:
            pass
        for column in self.numerical_fluctuating_columns:
            self.apply_univariate_imputation(column)
        for column in self.categorical_columns:
            pass
        missingl_perc = self.check_missing_val_perc()
        print(missingl_perc)

    def handle_outliers_timestamp(self):
        pass

    def handle_outliers_numerical_ordered(self):
        pass

    def handle_outliers_numerical_fluctuating(self):
        pass

    def handle_outliers(self):
        self.handle_outliers_timestamp()
        self.handle_outliers_numerical_ordered()
        self.handle_outliers_numerical_fluctuating()

    def process_data_cleaning(self):
        """
        Returns
        -------
        cleaned_df : pandas data frame
            Cleaned athlete CoachingMate data
        """
        self.handle_missing_values()
        self.handle_outliers()
        return self.dataframe


def make_cleaned_data_folder(data_type):
    if data_type == 'original':
        cleaned_original_folder = '{}/data/cleaned_original'.format(os.path.pardir)
        if not os.path.exists(cleaned_original_folder):
            os.mkdir(cleaned_original_folder)
    elif data_type == 'additional':
        cleaned_additional_folder = '{}/data/cleaned_additional'.format(os.path.pardir)
        if not os.path.exists(cleaned_additional_folder):
            os.mkdir(cleaned_additional_folder)
    else:
        raise Exception('No {} type of datasets'.format(data_type))


def main(data_type='original', athletes_name=None, activity_type=None, split_type=None):
    """Process the data cleaning

    Returns
    -------
    boolean
       Whether the data frame is empty
    """
    make_cleaned_data_folder(data_type)
    if data_type == 'original':
        data_loader_original = DataLoader('original')
        if athletes_name == None:
            # Clean all original data
            orig_file_names = utility.get_all_original_data_file_names()
            for file_name in orig_file_names:
                df = data_loader_original.load_original_data(file_name)
                original_data_cleaner = OriginalDataCleaner()
                cleaned_df = original_data_cleaner.process_data_cleaning(df)
                # cleaned_df.to_csv('{}/data/cleaned_original/{}'.format(os.path.pardir, file_name))
                # print('Cleaned {} data saved!'.format(file_name))
        else:
            # Clean data for an athlete
            df = data_loader_original.load_original_data(athletes_name)
            original_data_cleaner = OriginalDataCleaner()
            cleaned_df = original_data_cleaner.process_data_cleaning(df)
            file_name = data_loader_original.config.get('ORIGINAL-DATA-SETS', athletes_name.lower())
            # cleaned_df.to_csv('{}/data/cleaned_original/{}'.format(os.path.pardir, file_name))
            # print('Cleaned {} data saved!'.format(file_name))

    elif data_type == 'additional':
        data_loader_additional = DataLoader('additional')
        additional_file_names = data_loader_additional.load_additional_data(athletes_name=athletes_name,
                                                                 activity_type=activity_type,
                                                                 split_type=split_type)
        athlete_cleaned_additional_data_folder = '{}/data/cleaned_additional/{}'.format(os.path.pardir, athletes_name.lower())
        if not os.path.exists(athlete_cleaned_additional_data_folder):
            os.mkdir(athlete_cleaned_additional_data_folder)
        empty_files = []
        ## TODO: Remove acc
        acc = 0
        for file_name in additional_file_names:
            if acc < 2: acc += 1
            else: break
            print('Cleaning {} ...'.format(file_name[3:]))
            df = pd.DataFrame(pd.read_csv(file_name))
            addtional_data_cleaner = AdditionalDataCleaner(df)
            if addtional_data_cleaner.check_empty():
                empty_files.append(file_name)
            else:
                cleaned_df = addtional_data_cleaner.process_data_cleaning()
                # cleaned_df.to_csv('{}/{}'.format(athlete_cleaned_additional_data_folder, file_name[-41:]))
                # print('Cleaned {} data saved'.format(file_name[-41:]))

        print('For {}\'s additional data, {} out of {} {} files are empty.'.format(athletes_name,
                                                                                   len(empty_files),
                                                                                   len(additional_file_names),
                                                                                   activity_type))


if __name__ == '__main__':
    # Clean original data
    main('original')  # clean all original data
    main('original', 'eduardo oliveira')  # clean original data for one athlete

    # Clean additional data
    athletes_name = 'Eduardo Oliveira'
    activity_type = 'cycling'
    split_type = 'real-time'
    main('additional', athletes_name, activity_type, split_type)

