"""Data Cleaning

This script allows the user to process data cleaning for both CoachingMate data and Garmin data.

This script requires that `pandas`, `numpy` be installed within the Python
environment you are running this script in.

This file can also be imported as a module
"""

import numpy as np
import pandas as pd
import os
import datetime as dt
import matplotlib as plt
import utility
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from data_loader import DataLoader


import pandas_profiling

import warnings
warnings.filterwarnings('ignore')

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier)
pd.set_option('max.columns',100)
pd.set_option('max.rows',500)


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None

import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from IPython.display import Image
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
sns.set(style="darkgrid", palette="pastel", color_codes=True)
sns.set_context('talk')
from sklearn.impute import KNNImputer

import missingno as msno
from datetime import datetime


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

    def _drop_columns(self):
        # TODO: Choose columns instead of dropping them
        columns_to_drop = ['Favorite', 'Aerobic TE', 'Avg Run Cadence', 'Max Run Cadence', 'Avg Stride Length', 'Avg Vertical Ratio',
             'Avg Vertical Oscillation', 'Avg Ground Contact Time'
                , 'Avg GCT Balance', 'L/R Balance', 'Grit', 'Flow', 'Total Reps', 'Total Sets', 'Bottom Time',
             'Min Temp', 'Surface Interval', 'Decompression', 'Best Lap Time', 'Max Temp']
        # self.dataframe.drop(columns_to_drop, axis=1, inplace=True)

    def _convert_strings_to_lower_case(self):
        self.dataframe['Activity Type'] = self.dataframe['Activity Type'].str.lower()
        self.dataframe['Title'] = self.dataframe['Title'].str.lower()

    def _handle_commas(self):
        # TODO: Figure out for different columns, how do we treat comma. Is 1,000 1000 or 1.000?
        columns_remove_comma = self.dataframe.columns
        # columns_remove_comma = ['Max Avg Power (20 min)', 'Avg Power', 'Avg Stroke Rate', 'Avg HR', 'Max HR', 'Total Strokes',
        #            'Avg. Swolf', 'Avg Bike Cadence', 'Max Bike Cadence', 'Normalized Power® (NP®)',
        #            'Number of Laps']
        for column in columns_remove_comma:
            self.dataframe[column] = self.dataframe[column].astype(str).str.replace(',', '')
        # self.dataframe.apply(lambda x: x.str.replace(',', '.'))

    def _format_missing_val_with_nan(self):
        # TODO: Missing value situations in config and in functions to handle
        self.dataframe = self.dataframe.replace({ "--": np.nan, "...": np.nan })
        self.dataframe.loc[self.dataframe['Max Speed'].str.contains(":", na=False), 'Max Speed'] = np.nan
        self.dataframe.loc[self.dataframe['Avg Speed'].str.contains(":", na=False), 'Avg Speed'] = np.nan

    def _convert_columns_to_numeric(self):
        # TODO: columns in config
        columns_to_numeric = ['Max Avg Power (20 min)', 'Avg Power', 'Avg Stroke Rate', 'Avg HR', 'Max HR', 'Total Strokes',
                   'Avg. Swolf', 'Avg Bike Cadence', 'Max Bike Cadence', 'Normalized Power® (NP®)',
                   'Number of Laps']
        self.dataframe[columns_to_numeric] = self.dataframe[columns_to_numeric].apply(pd.to_numeric)

    def _convert_column_types_to_float(self):
        columns_to_float = ['Calories', 'Max Power', 'Max Speed', 'Avg Speed']
        for column in columns_to_float:
            self.dataframe[column].astype(float)

    def _format_datetime(self):
        self.dataframe['Date_extracted'] = pd.to_datetime(self.dataframe["Date"]).dt.normalize()
        self.dataframe['Time_extracted'] = pd.to_datetime(self.dataframe["Date"]).dt.time
        self.dataframe['Date'] = pd.to_datetime(self.dataframe['Date'])
        self.dataframe['Time_sec'] = pd.to_timedelta(
            pd.to_datetime(self.dataframe["Time"]).dt.strftime('%H:%M:%S')).dt.total_seconds()

    # handling irregular data
    # select numeric columns
    def get_numerical_columns(self):
        numeric_columns = self.dataframe.select_dtypes(include=[np.number])
        numeric_column_values = numeric_columns.columns.values
        return numeric_columns

    def get_categorical_columns(self):
        categorical_columns = self.dataframe.select_dtypes(exclude=[np.number])
        categoric_values = categorical_columns.columns.values
        return categorical_columns

    def find_missing_percent(self):
        """
        Returns dataframe containing the total missing values and percentage of total
        missing values of a column.
        """
        missing_val_df = pd.DataFrame({'ColumnName': [], 'TotalMissingVals': [], 'PercentMissing': []})
        for col in self.dataframe.columns:
            sum_miss_val = self.dataframe[col].isnull().sum()
            percent_miss_val = round((sum_miss_val / self.dataframe.shape[0]) * 100, 2)
            missing_val_df = missing_val_df.append(dict(zip(missing_val_df.columns, [col, sum_miss_val, percent_miss_val])),
                                         ignore_index=True)
        '''Columns with missing values'''
        print(f"Number of columns with missing values: {str(missing_val_df[missing_val_df['PercentMissing'] > 0.0].shape[0])}")
        return missing_val_df

    def plot_missing_val_bar(self):
        graph = msno.bar(self.dataframe)
        return graph

    def get_missingno_matrix(self):
        matrix = msno.matrix(self.dataframe)
        return matrix

    def make_heatmap(self):
        heatmap = msno.heatmap(self.dataframe)
        return heatmap

    def get_deno_gram(self):
        dendogram = msno.dendrogram(self.dataframe)
        return (dendogram)

    def get_profile_report(self):
        return pandas_profiling.ProfileReport(self.dataframe)

    def apply_mean_imputation(self, numeric_columns):
        for col in numeric_columns:
            mean = self.dataframe[col].mean()
            self.dataframe[col] = self.dataframe[col].fillna(mean)

    def process_data_cleaning(self):
        """
        Returns
        -------
        cleaned_df : pandas DataFrame
            Cleaned athlete CoachingMate data
        """
        self._drop_columns()
        self._convert_strings_to_lower_case()
        self._handle_commas()
        self._format_missing_val_with_nan()
        self._convert_columns_to_numeric()
        self._convert_column_types_to_float()
        self._format_datetime()
        numerical_cols = self.get_numerical_columns()
        self.apply_mean_imputation(numerical_cols)



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
        self.color_labels = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan', 'black',
                             'maroon', 'chocolate', 'gold', 'yellow', 'lawngreen', 'aqua', 'steelblue', 'navy',
                             'indigo',
                             'magenta', 'crimson', 'red']
        self.OUTLIER_COLOR_LABEL = len(self.color_labels) - 1
        self.TIME_DIFF_THRESHOLD = 3 * 60  # Configurable
        self.ROW_COUNT_THRESHOLD = 150  # Configurable
        self.num_recs = -1
        self.time_sgmt_num = 0

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
                print('{} has {}% missing'.format(column_name, round(missing_val_perc[column_name] * 100, 3)))
        if no_missing:
            print('Great! No missing values!')
        return missing_val_perc

    def get_columns_with_missing_values(self):
        cols_with_missing = {col for col in self.dataframe.columns if self.dataframe[col].isnull().any()}
        return cols_with_missing

    def convert_str_to_num_in_numerical_cols(self):
        for column in self.numerical_ordered_columns + self.numerical_fluctuating_columns:
            # dataframe[column] = dataframe[column].astype(float)
            self.dataframe[column] = pd.to_numeric(self.dataframe[column], errors='coerce')

    # def check_outliers(self):
    #     z = np.abs(stats.zscore(self.dataframe[self.numerical_ordered_columns]))
    #     threshold = 3
    #     outlier_zscores = np.where(z > threshold)
    #     outlier_rows_cols = outlier_zscores
    #     if len(outlier_zscores[0]) > 0:
    #         for row in outlier_zscores[0]:
    #             for col in outlier_zscores[1]:
    #                 outlier_zscore = z[row][col]
    #                 ourlier_value = self.dataframe.iloc[row, col]
    #     return outlier_rows_cols

    def add_column_time_in_seconds(self):
        time_in_seconds = []
        for time in self.dataframe['timestamp']:
            t = dt.datetime.strptime(time[:-6], '%Y-%m-%d %H:%M:%S')
            seconds = int(dt.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds())
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
        if null_cols:
            # If there are columns with all values missing, apply regression or ignore the column first.
            print("All the values in columns {} are missing. Not able to apply imputation.".format(null_cols))
            pass
        else:
            new_data = self.dataframe.copy()
            iter_imputer = IterativeImputer(max_iter=10, random_state=0)
            new_data = pd.DataFrame(iter_imputer.fit_transform(new_data[columns]))
            if not new_data.empty:
                new_data.columns = columns
                self.dataframe[columns] = new_data[columns]
            else:
                print("All the values in columns {} are missing. Not able to apply imputation.".format(columns))

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

    # def handle_outliers_timestamp(self):
    #     pass
    #
    # def handle_outliers_numerical_ordered(self):
    #     pass
    #
    # def handle_outliers_numerical_fluctuating(self):
    #     pass
    #
    # def handle_outliers(self):
    #     self.handle_outliers_timestamp()
    #     self.handle_outliers_numerical_ordered()
    #     self.handle_outliers_numerical_fluctuating()

    def _date_time_str_to_secs(self, dataframe):
        first_ts_datetime = dt.datetime.strptime(dataframe['timestamp'][0][:-6], "%Y-%m-%d %H:%M:%S")
        ts_secs = []
        for ts in dataframe['timestamp']:
            parsed_datetime = dt.datetime.strptime(ts[:-6], "%Y-%m-%d %H:%M:%S")
            ts_secs.append((parsed_datetime - first_ts_datetime).total_seconds())
        ts_secs = np.array(ts_secs)
        return ts_secs

    def _filter_by_repetition(self, dataframe, rec_color):
        df = dataframe.drop(columns=['timestamp', 'timestamp_utc', 'timezone'])
        dup_df = df[df.duplicated()]
        idx_list = dup_df.index.tolist()
        for i in idx_list:
            rec_color[i] = self.OUTLIER_COLOR_LABEL
            print("[OUTLIER DETECTED!] Index == %d Reason: DUPLICATE ROW!" % (i))

    def _filter_by_timestamp(self, ts_secs, OUTLIER_COLOR_LABEL, rec_color):
        # Some Initializations
        self.num_recs = len(ts_secs)
        self.time_sgmt_num = 0
        this_time_sgmt_row_cnt = 1

        i = 0
        for i in range(1, self.num_recs):
            secs_diff = ts_secs[i]-ts_secs[i-1]
            # Timestamps must be non-decreasing
            if (secs_diff < 0) :
                rec_color[i] = OUTLIER_COLOR_LABEL
                print("[OUTLIER DETECTED!] Index == %d Reason: DECREASING TIME STAMP!" % (i))

            # continuous training?
            elif (secs_diff >= self.TIME_DIFF_THRESHOLD):
                # Does the previous time segment have enough rows? If not, categorize all rows as outliers
                if (this_time_sgmt_row_cnt < self.ROW_COUNT_THRESHOLD):
                    for j in range(i-this_time_sgmt_row_cnt, i):

                        rec_color[j] = OUTLIER_COLOR_LABEL
                        print("[OUTLIER DETECTED!] Index == %d Reason: FEWER RECORDS ON THIS TIME SEGMENT(<%d)!" % (j, self.ROW_COUNT_THRESHOLD))

                # Setting up the new time segment
                self.time_sgmt_num += 1
                this_time_sgmt_row_cnt = 1
            else:
                this_time_sgmt_row_cnt += 1
                rec_color[i] = self.time_sgmt_num

        # Special treatment for the last time segment
        i += 1
        if (this_time_sgmt_row_cnt < self.ROW_COUNT_THRESHOLD):
            for j in range(i - this_time_sgmt_row_cnt, i):
                # -1 is the color number for outliers
                rec_color[j] = OUTLIER_COLOR_LABEL
                print("[OUTLIER DETECTED!] Index == %d Reason: FEWER RECORDS FOR THIS TIME SEGMENT(<%d)!" % (
                j, self.ROW_COUNT_THRESHOLD))

    def _filter_by_continuity_assumption(self, dataframe, rec_color):
        ## 1. The change of temperature should not exceed 1 degree
        for i in range(1, self.num_recs):
            if (rec_color[i] == rec_color[i-1] and rec_color[i] != self.OUTLIER_COLOR_LABEL):
                diff = abs(dataframe.loc[i, 'temperature'] - dataframe.loc[i-1, 'temperature'])
                if  diff > 1:
                    rec_color[i] = self.OUTLIER_COLOR_LABEL
                    print("[OUTLIER DETECTED!] Index == %d Reason: TEMPERATURE CHANGE TOO FAST!" % (i))

    def _filter_by_logical_inconsistency(self, dataframe, rec_color):
        ## 1. distance, enhanced_speed, speed, heart_rate and cadence must be greater than 0
        for i in range(self.num_recs):
            if dataframe['distance'][i] <= 0:
                rec_color[i] = self.OUTLIER_COLOR_LABEL
                print("[OUTLIER DETECTED!] Index == %d Reason: LOGICAL ERROR, distance has to be POSITIVE!" % (i))
            if dataframe['enhanced_speed'][i] <= 0:
                rec_color[i] = self.OUTLIER_COLOR_LABEL
                print("[OUTLIER DETECTED!] Index == %d Reason: LOGICAL ERROR, enhanced_speed has to be POSITIVE!" % (i))
            if dataframe['speed'][i] <= 0:
                rec_color[i] = self.OUTLIER_COLOR_LABEL
                print("[OUTLIER DETECTED!] Index == %d Reason: LOGICAL ERROR, speed has to be POSITIVE!" % (i))
            if dataframe['heart_rate'][i] <= 0:
                rec_color[i] = self.OUTLIER_COLOR_LABEL
                print("[OUTLIER DETECTED!] Index == %d Reason: LOGICAL ERROR, heart_rate has to be POSITIVE!" % (i))
            if dataframe['cadence'][i] <= 0:
                rec_color[i] = self.OUTLIER_COLOR_LABEL
                print("[OUTLIER DETECTED!] Index == %d Reason: LOGICAL ERROR, cadence has to be POSITIVE!" % (i))

    def _filter_by_zscore(self, dataframe, rec_color):
        # Configuraing different tolerace for different columns
        columns = ['position_lat', 'position_long', 'enhanced_altitude', 'altitude', 'heart_rate', 'cadence']
        tolerances = [2, 2, 3, 3, 3, 3]  # Configurable
        dataframe['color'] = rec_color
        for k in range(self.time_sgmt_num):
            df = dataframe.loc[dataframe['color'] == k]
            for j in range(len(columns)):
                # Actual Z-score test
                z = np.abs(stats.zscore(df[columns[j]]))
                outlier_rows = np.where(z > tolerances[j])
                print("\n====For Column:%s, There are %d outliers ===\n" % (columns[j], len(outlier_rows[0])))
                for i in outlier_rows[0]:
                    rec_color[i] = self.OUTLIER_COLOR_LABEL
                    print("[OUTLIER DETECTED!] Index == %d Reason: Z-SCORE(%f) EXCEEDS THE TOLERANCE(%f)" % (
                        i, z[i], tolerances[j]))

    def _plot_time_sgmt(self, ts_secs, colors):
        num_recs = len(ts_secs)
        zeros = np.zeros(num_recs).reshape(num_recs, 1)
        ts_secs = ts_secs.reshape(num_recs, 1)
        X = np.concatenate((ts_secs, zeros), axis=1)
        plt.scatter(X[:, 0], X[:, 1], c=colors)
        plt.show()

    def _cidx_to_clabels(self, rec_color):
        colors = []
        for i in range(len(rec_color)):
            colors.append(self.color_labels[rec_color[i]])

        return colors

    def _cidx_to_outlier_mask(self, rec_color):
        outlier_mask = []
        for i in range(self.num_recs):
            if rec_color[i] == self.OUTLIER_COLOR_LABEL:
                outlier_mask.append(1)
            else:
                outlier_mask.append(0)

        return outlier_mask

    # Returning (outlier_mask, df_outlier_free)
    # outlier_mask: 0 <==> not outlier; 1 <==> outlier
    # df_outlier_free: dataframe without possible outliers
    def handle_outliers(self):
        self.convert_str_to_num_in_numerical_cols()
        dataframe = self.dataframe
        self.num_recs = len(dataframe)

        # Some Initializations
        rec_color = np.zeros(self.num_recs, dtype=np.int32)

        # Convert time strings to epoch seconds
        ts_secs = self._date_time_str_to_secs(dataframe)

        # Filter outliers away by timestamp
        self._filter_by_timestamp(ts_secs, self.OUTLIER_COLOR_LABEL, rec_color)

        # Filter outliers away by removing repetitive rows
        self._filter_by_repetition(dataframe, rec_color)

        # Convert color index to labels
        colors = self._cidx_to_clabels(rec_color)

        # Plot Time Segments
        self._plot_time_sgmt(ts_secs, colors)

        # filter outliers away by continuity assumption
        self._filter_by_continuity_assumption(dataframe, rec_color)

        # filter outliers away by logical inconsistency
        self._filter_by_logical_inconsistency(dataframe, rec_color)

        # filter outliers away by z-scores
        self._filter_by_zscore(dataframe, rec_color)

        # Convert color index to outlier mask
        outlier_mask = self._cidx_to_outlier_mask(rec_color)

        # Assembly return value
        df_outlier_free = dataframe.loc[dataframe['color'] != self.OUTLIER_COLOR_LABEL]
        df_outlier_free = df_outlier_free.drop('color', 1)
        return (outlier_mask, df_outlier_free)

    def process_data_cleaning(self):
        """Process the data cleaning
        Returns
        -------
        cleaned_df : pandas data frame
            Cleaned athlete CoachingMate data
        """
        self.handle_missing_values()
        # self.handle_outliers()
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
            # file_name = data_loader_original.config.get('ORIGINAL-DATA-SETS', athletes_name.lower())
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
                print('File is empty.')
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
    # main('original')  # clean all original data
    main('original', 'eduardo oliveira')  # clean original data for one athlete

    # # Clean additional data
    # athletes_names = ['Eduardo Oliveira']
    # activity_type = ['cycling', 'running', 'swimming']
    # split_type = 'real-time'
    # main('additional', athletes_names[0], activity_type[0], split_type)  # clean all additional data with given condition

