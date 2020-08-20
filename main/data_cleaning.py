"""Data Cleaning

This script allows the user to process data cleaning for both CoachingMate data and Garmin data.

This script requires that `pandas`, `numpy` be installed within the Python
environment you are running this script in.

This file can also be imported as a module
"""

import numpy as np
import pandas as pd
import os
import datetime
import utility
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from data_loader import DataLoader
from configparser import ConfigParser

# Set the data frame display option
pd.set_option('display.max_row', 20)
pd.set_option('display.max_columns', 10)

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

    def __init__(self, dataframe: pd.DataFrame):
        self.numerical_columns = utility.get_original_numerical()
        self.categorical_columns = utility.get_original_categorical()
        self.dataframe = dataframe

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

    def clean_numerical_columns(self):
        pass

    def clean_categorical_columns(self):
        pass

    def process_data_cleaning(self):
        """
        Returns
        -------
        cleaned_df : pandas DataFrame
            Cleaned athlete CoachingMate data
        """
        self.clean_numerical_columns()
        self.clean_categorical_columns()
        return self.dataframe


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
        self.column_groups_imputation = {'univariate': {'timezone'},
                                         'multivariate1': {"distance", "timestamp"},
                                         'multivariate2': {"speed", "heart_rate", "cadence"},
                                         'interpolation': {"timestamp", "position_lat", "position_long", "altitude"}}
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
        no_missing = True
        missing_val_perc = self.dataframe.isnull().sum() / self.dataframe.shape[0]
        for column_name in missing_val_perc.keys():
            if float(missing_val_perc[column_name]) != 0:
                no_missing = False
                print('{} has {} missing'.format(column_name, missing_val_perc[column_name]))
        if no_missing:
            print('Great! No missing values!')
        return missing_val_perc

    def get_columns_with_missing_values(self):
        cols_with_missing = {col for col in self.dataframe.columns if self.dataframe[col].isnull().any()}
        return cols_with_missing

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

    def add_column_time_in_seconds(self):
        time_in_seconds = []
        for time in self.dataframe['timestamp']:
            t = datetime.datetime.strptime(time[:-6], '%Y-%m-%d %H:%M:%S')
            seconds = int(datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds())
            time_in_seconds.append(seconds)
        self.dataframe.insert(1, 'time_in_seconds', time_in_seconds, True)

    def apply_univariate_imputation(self, column):
        new_data = self.dataframe.copy()
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        new_data = pd.DataFrame(imputer.fit_transform(new_data[[column]]))
        new_data.columns = [column]
        self.dataframe[column] = new_data[column]

    def apply_multivariate_imputation(self, columns):
        null_cols = [col for col in columns if self.dataframe[col].isnull().all()]
        if not null_cols:
            new_data = self.dataframe.copy()
            iter_imputer = IterativeImputer(max_iter=10, random_state=0)
            new_data = pd.DataFrame(iter_imputer.fit_transform(new_data[columns]))
            if not new_data.empty:
                new_data.columns = columns
                self.dataframe[columns] = new_data[columns]
            else:
                print("All the values in columns {} are missing. Not able to apply imputation.".format(columns))
        else:
            # TODO: Apply regression or ignore
            pass

    def apply_interpolation_imputation(self, columns):
        for column in columns:
            interpolated_column = self.dataframe[column].interpolate(method='spline', order=2, limit=2)
            self.dataframe[column] = interpolated_column

    def apply_nearest_neighbor_imputation(self, missing_values=np.nan, strategy="mean"):
        pass

    def apply_regression_prediction_interpolation(self, column_names):
        # TODO: for whole column missing
        pass

    def apply_imputations(self, columns_need_imputation):
        for impute_tech, column_names in self.column_groups_imputation.items():
            column_intersection = column_names.intersection(columns_need_imputation)
            if column_intersection:
                if impute_tech == "univariate":
                    self.apply_univariate_imputation(list(column_intersection))
                elif impute_tech == "multivariate1" or "multivariate2":
                    self.apply_multivariate_imputation(list(column_intersection))
                elif impute_tech == "interpolation":
                    self.apply_regression_prediction_interpolation(list(column_intersection))

    def handle_missing_values(self):
        self.format_missing_val_with_nan()
        self.check_missing_val_perc()
        columns_need_imputation = self.get_columns_with_missing_values()
        if columns_need_imputation:
            self.add_column_time_in_seconds()
            self.apply_imputations(columns_need_imputation)
        print('After imputation: ')
        self.check_missing_val_perc()

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
    """The main function of processing data cleaning
    Clean the data and save the cleaned data

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
                original_data_cleaner = OriginalDataCleaner(df)
                cleaned_df = original_data_cleaner.process_data_cleaning()
                # cleaned_df.to_csv('{}/data/cleaned_original/{}'.format(os.path.pardir, file_name))
                # print('Cleaned {} data saved!'.format(file_name))
        else:
            # Clean data for an athlete
            df = data_loader_original.load_original_data(athletes_name)
            original_data_cleaner = OriginalDataCleaner(df)
            cleaned_df = original_data_cleaner.process_data_cleaning()
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
        for file_name in additional_file_names:
            print('\nCleaning {} ...'.format(file_name[3:]))
            df = pd.DataFrame(pd.read_csv(file_name))
            addtional_data_cleaner = AdditionalDataCleaner(df)
            if addtional_data_cleaner.check_empty():
                empty_files.append(file_name)
            else:
                cleaned_df = addtional_data_cleaner.process_data_cleaning()
                # cleaned_df.to_csv('{}/{}'.format(athlete_cleaned_additional_data_folder, file_name[-41:]))
                # print('Cleaned {} data saved'.format(file_name[-41:]))

        print('\nFor {}\'s additional data, {} out of {} {} files are empty.'.format(athletes_name,
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
    main('additional', athletes_name, activity_type, split_type)  # clean all additional data

