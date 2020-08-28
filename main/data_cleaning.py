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
import xgboost
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import (GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.preprocessing import MinMaxScaler

import matplotlib.mlab as mlab
import matplotlib.pyplot as pyplot
import matplotlib

pyplot.style.use('ggplot')
from matplotlib.pyplot import figure

matplotlib.rcParams['figure.figsize'] = (12, 8)

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


class OriginalDataCleaner() :
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

    def __init__(self, dataframe: pd.DataFrame) :
        # self.numerical_columns = utility.get_original_numerical()
        # self.categorical_columns = utility.get_original_categorical()
        self.dataframe = dataframe

    def _drop_columns(self) :
        # TODO: Choose columns instead of dropping them
        columns_to_drop = ['Favorite', 'Aerobic TE', 'Avg Run Cadence', 'Max Run Cadence', 'Avg Stride Length',
                           'Avg Vertical Ratio', 'Avg Vertical Oscillation', 'Avg Ground Contact Time',
                           'Avg GCT Balance', 'L/R Balance', 'Grit', 'Flow', 'Total Reps', 'Total Sets',
                           'Bottom Time', 'Min Temp', 'Surface Interval', 'Decompression', 'Best Lap Time', 'Max Temp']
        self.dataframe.drop(columns_to_drop, axis=1, inplace=True)

    def _convert_strings_to_lower_case(self) :
        self.dataframe['Activity Type'] = self.dataframe['Activity Type'].str.lower()
        self.dataframe['Title'] = self.dataframe['Title'].str.lower()

    def _handle_commas(self) :
        # TODO: Figure out for different columns, how do we treat comma. Is 1,000 1000 or 1.000?
        columns_remove_comma = self.dataframe.columns
        # columns_remove_comma = ['Max Avg Power (20 min)', 'Avg Power', 'Avg Stroke Rate', 'Avg HR', 'Max HR', 'Total Strokes',
        #            'Avg. Swolf', 'Avg Bike Cadence', 'Max Bike Cadence', 'Normalized Power® (NP®)',
        #            'Number of Laps']
        for column in columns_remove_comma :
            self.dataframe[column] = self.dataframe[column].astype(str).str.replace(',', '')
            # TODO: Handling semicolon
            # self.dataframe[column] = self.dataframe[column].astype(str).str.replace(':', '')
        # self.dataframe.apply(lambda x: x.str.replace(',', '.'))

    def _format_missing_val_with_nan(self) :
        # TODO: Missing value situations in config and in functions to handle
        self.dataframe = self.dataframe.replace({"--" : np.nan, "..." : np.nan})
        self.dataframe.loc[self.dataframe['Max Speed'].str.contains(":", na=False), 'Max Speed'] = np.nan
        self.dataframe.loc[self.dataframe['Avg Speed'].str.contains(":", na=False), 'Avg Speed'] = np.nan

    def _convert_columns_to_numeric(self) :
        # TODO: columns in config
        columns_to_numeric = ['Max Avg Power (20 min)', 'Avg Power', 'Avg Stroke Rate', 'Avg HR', 'Max HR',"Distance",'Training Stress Score®',
                              'Total Strokes','Elev Gain', 'Elev Loss','Calories', 'Max Power','Max Speed', 'Avg Speed'
                              ,'Avg. Swolf', 'Avg Bike Cadence', 'Max Bike Cadence', 'Normalized Power® (NP®)',
                              'Number of Laps']
        self.dataframe[columns_to_numeric] = self.dataframe[columns_to_numeric].apply(pd.to_numeric)

    # def _convert_column_types_to_float(self):
    # TODO:fred for some reason this fucntion is not working,is htere any diff to above
    #     columns_to_float = [ 'Max Speed', 'Avg Speed']
    #     for column in columns_to_float:
    #         #self.dataframe[column].astype(float)
    #          self.dataframe[column].apply(pd.to_numeric)
    #
    #     #print(self.dataframe["Max Power"].dtypes())

    def _format_datetime(self):
        self.dataframe['Date_extracted'] = pd.to_datetime(self.dataframe["Date"]).dt.normalize()
        self.dataframe['Time_extracted'] = pd.to_datetime(self.dataframe["Date"]).dt.time
        self.dataframe['Date'] = pd.to_datetime(self.dataframe['Date'])
        self.dataframe['Time_sec'] = pd.to_timedelta(
            pd.to_datetime(self.dataframe["Time"]).dt.strftime('%H:%M:%S')).dt.total_seconds()

    def find_missing_percent(self):
        """
        Returns dataframe containing the total missing values and percentage of total
        missing values of a column.
        """
        missing_val_df = pd.DataFrame({'ColumnName' : [], 'TotalMissingVals' : [], 'PercentMissing' : []})
        for col in self.dataframe.columns :
            sum_miss_val = self.dataframe[col].isnull().sum()
            percent_miss_val = round((sum_miss_val / self.dataframe.shape[0]) * 100, 2)
            missing_val_df = missing_val_df.append(
                dict(zip(missing_val_df.columns, [col, sum_miss_val, percent_miss_val])),
                ignore_index=True)
        '''Columns with missing values'''
        print(
            f"Number of columns with missing values: {str(missing_val_df[missing_val_df['PercentMissing'] > 0.0].shape[0])}")
        return missing_val_df

    def plot_missing_val_bar(self) :
        graph = msno.bar(self.dataframe)
        return graph

    def get_missingno_matrix(self) :
        matrix = msno.matrix(self.dataframe)
        return matrix

    def make_heatmap(self) :
        heatmap = msno.heatmap(self.dataframe)
        return heatmap

    def get_deno_gram(self) :
        dendogram = msno.dendrogram(self.dataframe)
        return (dendogram)

    def get_profile_report(self) :
        return pandas_profiling.ProfileReport(self.dataframe)

    # handling irregular data
    # select numeric columns
    def get_numerical_columns(self) :
        numeric_column_df = self.dataframe.select_dtypes(include=[np.number])
        numeric_column_values = numeric_column_df.columns.values
        return numeric_column_df, numeric_column_values

    def get_categorical_columns(self) :
        categorical_columns = self.dataframe.select_dtypes(exclude=[np.number])
        categoric_values = categorical_columns.columns.values
        return categorical_columns, categoric_values

    def _apply_mean_imputation(self, data_numeric) :
        """Apply mean imputation for the given columns
        # TODO: A reminder to Sindhu: For this function, input is a list of column names which are strings,
        # TODO: no output, just modify self.dataframe.
        # TODO: So after calling this function, self.dataframe has the specified columns imputated.
        #Parameters
        #-------
        #columns: [str]
           #List of column names
        #"""
        for col in data_numeric.columns :
            mean = data_numeric[col].mean()
            data_numeric[col] = data_numeric[col].fillna(mean)
        return data_numeric

    def _find_missing_index(self, data_numeric_regr, target_cols) :
        miss_index_dict = {}
        for tcol in target_cols :
            index = data_numeric_regr[tcol][data_numeric_regr[tcol].isnull()].index
            miss_index_dict[tcol] = index
        return miss_index_dict

    def _apply_regression_imputation(self, data_numeric_regr, target_cols, miss_index_dict) :
        '''Predictors for regression imputation'''
        predictors = data_numeric_regr.drop(target_cols, axis=1)
        for tcol in target_cols :
            y = data_numeric_regr[tcol]
            '''Initially impute the column with mean'''
            y = y.fillna(y.mean())
            xgb = xgboost.XGBRegressor(objective="reg:squarederror", random_state=42)
            '''Fit the model where y is the target column which is to be imputed'''
            xgb.fit(predictors, y)
            predictions = pd.Series(xgb.predict(predictors), index=y.index)
            index = miss_index_dict[tcol]
            '''Replace the missing values with the predictions'''
            data_numeric_regr[tcol].loc[index] = predictions.loc[index]
        return data_numeric_regr

    def _apply_linear_interpolation(self, numeric_column_df):
        for col in numeric_column_df.columns:
            numeric = numeric_column_df.interpolate(method='linear', limit_direction='forward', axis=0).ffill().bfill()
        return (numeric)

    def _apply_mice_imputation_numeric(self, numeric_column_df):
        iter_imp_numeric = IterativeImputer(GradientBoostingRegressor())
        imputed_eddy = iter_imp_numeric.fit_transform(numeric_column_df)
        eddy_numeric_imp = pd.DataFrame(imputed_eddy, columns=numeric_column_df.columns, index=numeric_column_df.index)
        return eddy_numeric_imp

    def get_minmax(self, numeric_column_df, numeric_column_values):
        'when imputing a knn data must be normalised to reduce the bias in the imoutation'
        scaler = MinMaxScaler()
        scaling = pd.DataFrame(scaler.fit_transform(numeric_column_df), columns=numeric_column_values)
        return scaling

    def _apply_knn_imputation(self, numeric_column_df, numeric_column_values):
        imputer = KNNImputer(n_neighbors=23)
        imputed_KNN = pd.DataFrame(imputer.fit_transform(numeric_column_df), columns=numeric_column_values)
        return imputed_KNN

    def _apply_mode_imputation(self,categorical_columns):
        """
        Mode Imputation
        """
        for col in categorical_columns.columns :
            mode = categorical_columns[col].mode().iloc[0]
            categorical_columns[col] = categorical_columns[col].fillna(mode)
        return categorical_columns



    # def mice_imputation_categoric(self,categorical_columns):
    #     ordinal_dict = {}
    #     for col in categorical_columns.columns:
    #         ordinal_dict[col] = OrdinalEncoder()
    #         nn_vals = np.array(categorical_columns[col][categorical_columns[col].notnull()]).reshape(-1, 1)
    #         nn_vals_arr = np.array(ordinal_dict[col].fit_transform(nn_vals)).reshape(-1, )
    #         categorical_columns[col].loc[categorical_columns[col].notnull()] = nn_vals_arr
    #     '''Impute the data using MICE with Gradient Boosting Classifier'''
    #     iter_imp_categoric = IterativeImputer(GradientBoostingClassifier(), max_iter=5,
    #                                           initial_strategy='most_frequent')
    #     imputed_eddy = iter_imp_categoric.fit_transform(categorical_columns)
    #     eddy_categoric_imp = pd.DataFrame(imputed_eddy, columns=categorical_columns.columns,
    #                                       index=categorical_columns.index).astype(int)
    #     '''Inverse Transform'''
    #     for col in eddy_categoric_imp.columns:
    #         oe = ordinal_dict[col]
    #         eddy_arr = np.array(eddy_categoric_imp[col]).reshape(-1, 1)
    #         eddy_categoric_imp[col] = oe.inverse_transform(eddy_arr)
    #     return eddy_categoric_imp
    def out_iqr(self, column):
        global lower, upper
        q25, q75 = np.quantile(self.dataframe[column], 0.25), np.quantile(self.dataframe[column], 0.75)
        # calculate the IQR
        iqr = q75 - q25
        # calculate the outlier cutoff
        cut_off = iqr * 1.5
        # calculate the lower and upper bound value
        lower, upper = q25 - cut_off, q75 + cut_off
        print('The IQR is', iqr)
        print('The lower bound value is', lower)
        print('The upper bound value is', upper)
        # Calculate the number of records below and above lower and above bound value respectively
        df1 = self.dataframe[self.dataframe[column] > upper]
        df2 = self.dataframe[self.dataframe[column] < lower]
        return print('Total number of outliers are', df1.shape[0] + df2.shape[0])





    def process_data_cleaning(self) :
        """
        Returns
        -------
        cleaned_df : pandas DataFrame
            Cleaned athlete CoachingMate data
        """
        # TODO: A reminder to Sindhu: MAIN FUNCTION FOR THE CLASS
        # TODO: Everytime you finish a function, call it below, and test below.
        self._drop_columns()
        self._convert_strings_to_lower_case()
        self._handle_commas()
        self._format_missing_val_with_nan()
        self._convert_columns_to_numeric()

        #self._convert_column_types_to_float()
        #print(self._convert_column_types_to_float())
        self._format_datetime()
        self._convert_columns_to_numeric()
        numeric_column_df, numeric_column_values = self.get_numerical_columns()
        #print(self.get_numerical_columns())
        categorical_columns, categoric_values = self.get_categorical_columns()
        #print(self.get_categorical_columns())
        data_numeric = self.dataframe[numeric_column_values]
        self._apply_mean_imputation(data_numeric)
        data_numeric_regr = self.dataframe[numeric_column_values]
        # # # # # '''Numeric columns with missing values which acts as target in training'''
        target_cols = ['Avg. Swolf','Total Strokes','Avg Speed','Avg HR', 'Max HR', 'Avg Bike Cadence', 'Max Bike Cadence',"Distance","Calories","Avg Stroke Rate","Number of Laps"]
        predictors = data_numeric_regr.drop(target_cols, axis=1)
        miss_index_dict = self._find_missing_index(data_numeric_regr, target_cols)
        self._apply_regression_imputation(data_numeric_regr, target_cols, miss_index_dict)
        #print(self._apply_regression_imputation(data_numeric_regr, target_cols, miss_index_dict))
        self._apply_linear_interpolation(numeric_column_df)
        eddy_numeric_imp = self._apply_mice_imputation_numeric(numeric_column_df)
        self.get_minmax(numeric_column_df, numeric_column_values)
        self.dataframe=self._apply_knn_imputation(numeric_column_df, numeric_column_values)#assigning imputaion can change later
        self._apply_mode_imputation(categorical_columns)
        # self.mice_imputation_categoric(categorical_columns)
        print(self.dataframe.isna().any())
        print(self.dataframe.isna().sum())
        self.out_iqr("Distance")



class AdditionalDataCleaner() :
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

    def __init__(self, dataframe: pd.DataFrame, file_name: str = None) :
        self.dataframe = dataframe
        self.file_name = file_name
        self.column_groups_imputation = {'univariate' : {'timezone'},
                                         'multivariate1' : {"distance", "timestamp"},
                                         'multivariate2' : {"speed", "heart_rate", "cadence"},
                                         'interpolation' : {"timestamp", "position_lat", "position_long", "altitude"}}
        self.numerical_ordered_columns = utility.get_additional_numerical_ordered()
        self.numerical_fluctuating_columns = utility.get_additional_numerical_fluctuating()
        self.categorical_columns = utility.get_additional_categorical()
        self.missing_val_logger = []
        self.outlier_dict_logger = {'Outlier Index' : [], 'Column' : [], 'Reason' : []}

        self.color_labels = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan', 'black',
                             'maroon', 'chocolate', 'gold', 'yellow', 'lawngreen', 'aqua', 'steelblue', 'navy',
                             'indigo', 'magenta', 'crimson', 'red']
        self.OUTLIER_COLOR_LABEL = len(self.color_labels) - 1
        self.TIME_DIFF_THRESHOLD = 3 * 60  # Configurable
        self.ROW_COUNT_THRESHOLD = 150  # Configurable
        self.num_records = -1
        self.time_sgmt_num = 0

    def check_empty(self) :
        """Check whether the data frame is empty

        Returns
        -------
        boolean
           Whether the data frame is empty
        """
        return self.dataframe.empty

    def format_missing_val_with_nan(self) :
        self.dataframe.replace('None', np.nan)

    def check_missing_val_perc(self) :
        no_missing = True
        missing_val_perc = self.dataframe.isnull().sum() / self.dataframe.shape[0]
        for column_name in missing_val_perc.keys() :
            if float(missing_val_perc[column_name]) != 0 :
                no_missing = False
                self.missing_val_logger.append(
                    '{} has {}% missing. \n'.format(column_name, round(missing_val_perc[column_name] * 100, 3)))
        if no_missing :
            self.missing_val_logger.append('Great! No missing values! \n')
        return missing_val_perc

    def get_columns_with_missing_values(self) :
        cols_with_missing = {col for col in self.dataframe.columns if self.dataframe[col].isnull().any()}
        return cols_with_missing

    def convert_str_to_num_in_numerical_cols(self) :
        for column in self.numerical_ordered_columns + self.numerical_fluctuating_columns :
            # dataframe[column] = dataframe[column].astype(float)
            self.dataframe[column] = pd.to_numeric(self.dataframe[column], errors='coerce')

    def add_column_time_in_seconds(self) :
        time_in_seconds = []
        for time in self.dataframe['timestamp'] :
            t = dt.datetime.strptime(time[:-6], '%Y-%m-%d %H:%M:%S')
            seconds = int(dt.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds())
            time_in_seconds.append(seconds)
        self.dataframe.insert(1, 'time_in_seconds', time_in_seconds, True)

    def _apply_univariate_imputation(self, columns) :
        new_data = self.dataframe.copy()
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        new_data = pd.DataFrame(imputer.fit_transform(new_data[[columns]]))
        new_data.columns = [columns]
        self.dataframe[columns] = new_data[columns]

    def _apply_multivariate_imputation(self, columns) :
        null_cols = [col for col in columns if self.dataframe[col].isnull().all()]
        if null_cols :
            # If there are columns with all values missing, apply regression or ignore the column first.
            self.missing_val_logger.append("All the values in column(s) {} are missing. "
                                           "Not able to apply imputation.".format(null_cols))
        else :
            new_data = self.dataframe.copy()
            iter_imputer = IterativeImputer(max_iter=10, random_state=0)
            new_data = pd.DataFrame(iter_imputer.fit_transform(new_data[columns]))
            if not new_data.empty :
                new_data.columns = columns
                self.dataframe[columns] = new_data[columns]
            else :
                self.missing_val_logger.append("All the values in column(s) {} are missing. "
                                               "Not able to apply imputation.".format(columns))

    def _apply_interpolation_imputation(self, columns) :
        for column in columns :
            interpolated_column = self.dataframe[column].interpolate(method='spline', order=2, limit=2)
            self.dataframe[column] = interpolated_column

    def _apply_regression_prediction_imputation(self, column_names) :
        # TODO: for whole column missing
        pass

    def _apply_imputations(self, columns_need_imputation) :
        for impute_tech, column_names in self.column_groups_imputation.items() :
            column_intersection = column_names.intersection(columns_need_imputation)
            if column_intersection :
                if impute_tech == "univariate" :
                    self._apply_univariate_imputation(list(column_intersection))
                elif impute_tech == "multivariate1" or "multivariate2" :
                    self._apply_multivariate_imputation(list(column_intersection))
                elif impute_tech == "interpolation" :
                    self._apply_regression_prediction_imputation(list(column_intersection))

    def handle_missing_values(self) :
        self.format_missing_val_with_nan()
        self.missing_val_logger.append('Before imputation: \n')
        self.check_missing_val_perc()
        columns_need_imputation = self.get_columns_with_missing_values()
        if columns_need_imputation :
            self._apply_imputations(columns_need_imputation)
        self.missing_val_logger.append('After imputation: \n')
        self.check_missing_val_perc()

    def _log_outlier(self, outlier_index, column, reason) :
        self.outlier_dict_logger['Outlier Index'].append(outlier_index)
        self.outlier_dict_logger['Column'].append(column)
        self.outlier_dict_logger['Reason'].append(reason)

    def _filter_by_repetition(self, rec_color) :
        df = self.dataframe.drop(columns=['timestamp', 'timestamp_utc', 'timezone'])
        dup_df = df[df.duplicated()]
        idx_list = dup_df.index.tolist()
        for i in idx_list :
            rec_color[i] = self.OUTLIER_COLOR_LABEL
            self._log_outlier(i, 'ALL', 'DUPLICATE ROW')

    def _filter_by_timestamp(self, rec_color) :
        # Convert time strings to epoch seconds
        ts_secs = np.array(self.dataframe['time_in_seconds'])
        self.num_records = len(ts_secs)
        self.time_sgmt_num = 0
        this_time_sgmt_row_cnt = 1

        i = 0
        for i in range(1, self.num_records) :
            secs_diff = ts_secs[i] - ts_secs[i - 1]
            # Timestamps must be non-decreasing
            if (secs_diff < 0) :
                rec_color[i] = self.OUTLIER_COLOR_LABEL
                self._log_outlier(i, 'timestamp', 'DECREASING TIME STAMP')

            # Check if it is a continuous training
            elif (secs_diff >= self.TIME_DIFF_THRESHOLD) :
                # If the previous time segment does not have enough rows, categorize all rows as outliers
                if (this_time_sgmt_row_cnt < self.ROW_COUNT_THRESHOLD) :
                    for j in range(i - this_time_sgmt_row_cnt, i) :
                        rec_color[j] = self.OUTLIER_COLOR_LABEL
                        self._log_outlier(i, 'timestamp',
                                          'FEWER RECORDS ON THIS TIME SEGMENT(<%d)!' % (self.ROW_COUNT_THRESHOLD))

                # Setting up the new time segment
                self.time_sgmt_num += 1
                this_time_sgmt_row_cnt = 1
            else :
                this_time_sgmt_row_cnt += 1
                rec_color[i] = self.time_sgmt_num

        # Special treatment for the last time segment
        i += 1
        if (this_time_sgmt_row_cnt < self.ROW_COUNT_THRESHOLD) :
            for j in range(i - this_time_sgmt_row_cnt, i) :
                # -1 is the color number for outliers
                rec_color[j] = self.OUTLIER_COLOR_LABEL
                self._log_outlier(j, 'timestamp',
                                  'FEWER RECORDS FOR THIS TIME SEGMENT(<%d)!' % (self.ROW_COUNT_THRESHOLD))

    def _filter_by_continuity_assumption(self, i, rec_color) :
        ## The change of temperature should not exceed 1 degree

        if (rec_color[i] == rec_color[i - 1] and rec_color[i] != self.OUTLIER_COLOR_LABEL) :
            diff = abs(self.dataframe.loc[i, 'temperature'] - self.dataframe.loc[i - 1, 'temperature'])
            if diff > 1 :
                rec_color[i] = self.OUTLIER_COLOR_LABEL
                self._log_outlier(i, 'temperature', 'TEMPERATURE CHANGES TOO FAST')
                # print("[OUTLIER DETECTED!] Index == %d Reason: TEMPERATURE CHANGES TOO FAST!" % (i))

    def _filter_by_logical_inconsistency(self, i, rec_color) :
        # distance, enhanced_speed, speed, heart_rate and cadence must be greater than or equal to 0
        # for i in range(self.num_records):
        for column in ['distance', 'enhanced_speed', 'speed', 'heart_rate', 'cadence'] :
            if self.dataframe[column][i] < 0 :
                # take the zero values as missing values
                rec_color[i] = self.OUTLIER_COLOR_LABEL
                self._log_outlier(i, column, 'LOGICAL ERROR, {} has to be non-negative!'.format(column))

    def _filter_by_zscore(self, dataframe, rec_color) :
        # Configuraing different tolerace for different columns
        columns = ['position_lat', 'position_long', 'enhanced_altitude', 'altitude', 'heart_rate', 'cadence']
        tolerances = [2, 2, 3, 3, 3, 3]  # Configurable
        dataframe['color'] = rec_color
        for k in range(self.time_sgmt_num) :
            df = dataframe.loc[dataframe['color'] == k]
            for j in range(len(columns)) :
                # Actual Z-score test
                z = np.abs(stats.zscore(df[columns[j]]))
                outlier_rows = np.where(z > tolerances[j])
                # print("\n====For Column:%s, There are %d outliers ===\n" % (columns[j], len(outlier_rows[0])))
                for i in outlier_rows[0] :
                    rec_color[i] = self.OUTLIER_COLOR_LABEL
                    self._log_outlier(i, columns[j],
                                      'Z-SCORE({}) EXCEEDS THE TOLERANCE ({})'.format(z[i], tolerances[j]))

    def _boxplot(self, df, rec_color) :
        """ Boxplot for the outliers
        :param df:
        :param rec_color:
        :return:
        """
        pyplot.boxplot(df, sym="o", whis=1.5)
        pyplot.show()
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        IQR_predicate = (df < Q1 - 1.5 * IQR) | (df > Q3 + 1.5 * IQR)
        dropped = df[IQR_predicate]
        for idx in dropped :
            rec_color[idx] = self.OUTLIER_COLOR_LABEL

    def _plot_time_sgmt(self, colors) :
        ts_secs = np.array(self.dataframe['time_in_seconds'])
        num_recs = len(ts_secs)
        zeros = np.zeros(num_recs).reshape(num_recs, 1)
        ts_secs = ts_secs.reshape(num_recs, 1)
        X = np.concatenate((ts_secs, zeros), axis=1)
        pyplot.scatter(X[:, 0], X[:, 1], c=colors)
        pyplot.show()

    def _cidx_to_clabels(self, rec_color) :
        colors = []
        for i in range(len(rec_color)) :
            colors.append(self.color_labels[rec_color[i]])
        return colors

    def _cidx_to_outlier_mask(self, rec_color) :
        outlier_mask = [1 if rec_color[i] == self.OUTLIER_COLOR_LABEL else 0 for i in range(self.num_records)]
        return outlier_mask

    def get_outliers_by_zscore(self) :
        z = np.abs(stats.zscore(self.dataframe[self.numerical_ordered_columns]))
        threshold = 3
        outlier_zscores = np.where(z > threshold)
        outlier_rows_cols = outlier_zscores
        if len(outlier_zscores[0]) > 0 :
            for row in outlier_zscores[0] :
                for col in outlier_zscores[1] :
                    outlier_zscore = z[row][col]
                    ourlier_value = self.dataframe.iloc[row, col]
        return outlier_rows_cols

    def get_outliers_timestamp(self) :
        pass

    def handle_outliers_numerical_ordered(self) :
        pass

    def handle_outliers_numerical_fluctuating(self) :
        pass

    def handle_outliers(self) :
        """

        :return: (outlier_mask, df_outlier_free)
            outlier_mask: 0 <==> not outlier; 1 <==> outlier
            df_outlier_free: dataframe without possible outliers
        """
        self.convert_str_to_num_in_numerical_cols()
        dataframe = self.dataframe
        self.num_records = len(dataframe)

        # Some Initializations
        rec_color = np.zeros(self.num_records, dtype=np.int32)

        # Filter outliers away by removing repetitive rows
        self._filter_by_repetition(rec_color)

        # # Filter outliers away by timestamp
        # self._filter_by_timestamp(rec_color)
        # # Convert color index to labels
        # colors = self._cidx_to_clabels(rec_color)
        # # Plot Time Segments
        # self._plot_time_sgmt(colors)

        for i in range(1, self.num_records) :
            # filter outliers away by continuity assumption
            self._filter_by_continuity_assumption(i, rec_color)
            # filter outliers away by logical inconsistency
            self._filter_by_logical_inconsistency(i, rec_color)

        # filter outliers away by z-scores
        self._filter_by_zscore(dataframe, rec_color)

        # Convert color index to outlier mask
        outlier_mask = self._cidx_to_outlier_mask(rec_color)

        # Assembly return value
        df_outlier_free = dataframe.loc[dataframe['color'] != self.OUTLIER_COLOR_LABEL]
        df_outlier_free = df_outlier_free.drop('color', 1)
        return (outlier_mask, df_outlier_free)

    def process_data_cleaning(self) :
        """Process the data cleaning
        Returns
        -------
        cleaned_df : pandas data frame
            Cleaned athlete CoachingMate data
        """
        self.add_column_time_in_seconds()
        self.handle_missing_values()
        self.handle_outliers()
        return self.dataframe


def _create_cleaned_data_folder(data_type) :
    if data_type == 'original' :
        cleaned_original_folder = '{}/data/cleaned_original'.format(os.path.pardir)
        if not os.path.exists(cleaned_original_folder) :
            os.mkdir(cleaned_original_folder)
    elif data_type == 'additional' :
        cleaned_additional_folder = '{}/data/cleaned_additional'.format(os.path.pardir)
        if not os.path.exists(cleaned_additional_folder) :
            os.mkdir(cleaned_additional_folder)
    else :
        raise Exception('No {} type of datasets'.format(data_type))


def _create_log_folders(data_type, athletes_name=None) :
    if not os.path.exists('{}/log/'.format(os.path.pardir)) :
        os.mkdir('{}/log/'.format(os.path.pardir))

    if data_type == 'original' :
        log_folder_names = ['original_missing_value_log',
                            'original_outlier_log']
    elif data_type == 'additional' :
        log_folder_names = ['additional_missing_value_log',
                            'additional_outlier_log']
        if athletes_name :
            log_folder_names.extend(['{}/{}'.format(name, athletes_name) for name in log_folder_names])
    else :
        raise Exception('No {} type of datasets'.format(data_type))

    for log_folder_name in log_folder_names :
        folder = '{}/log/{}'.format(os.path.pardir, log_folder_name)
        if not os.path.exists(folder) :
            os.mkdir(folder)


def _save_cleaned_df(data_type, athletes_name, file_name, cleaned_df) :
    if data_type == 'original' :
        cleaned_df.to_csv('{}/data/cleaned_original/{}'.format(os.path.pardir, file_name))
        print('Cleaned {} data saved!'.format(file_name))

    elif data_type == 'additional' :
        athlete_cleaned_additional_data_folder = '{}/data/cleaned_additional/{}'.format(os.path.pardir,
                                                                                        athletes_name.lower())
        if not os.path.exists(athlete_cleaned_additional_data_folder) :
            os.mkdir(athlete_cleaned_additional_data_folder)
        cleaned_df.to_csv('{}/{}'.format(athlete_cleaned_additional_data_folder, file_name[-41 :]))
        print('{}\'s cleaned {} data saved'.format(athletes_name.capitalize(), file_name[-41 :]))


def _save_log(data_type, log_type, file_name, log_df, athletes_name=None) :
    if data_type == 'original' :
        log_file_path = '{}/log/{}_{}_log/{}'.format(os.path.pardir, data_type, log_type, file_name[-41 :])
    elif data_type == 'additional' :
        log_file_path = '{}/log/{}_{}_log/{}/{}'.format(os.path.pardir, data_type, log_type, athletes_name,
                                                        file_name[-41 :])
    else :
        raise Exception('No {} type of datasets'.format(data_type))
    if not log_df.empty :
        log_df.to_csv(log_file_path)


def _main_helper_original(athletes_name=None, file_name: str = None) :
    # TODO: Create missing value and outlier log for original data
    data_loader_original = DataLoader('original')
    if file_name :
        athlete_df = data_loader_original.load_original_data(file_name)
    else :
        athlete_df = data_loader_original.load_original_data(athletes_name)
        file_name = data_loader_original.config.get('ORIGINAL-DATA-SETS', athletes_name.lower())
    original_data_cleaner = OriginalDataCleaner(athlete_df)
    original_data_cleaner.process_data_cleaning()
    cleaned_df = original_data_cleaner.dataframe
    _save_cleaned_df('original', athletes_name, file_name, cleaned_df)


def _main_helper_additional(athletes_name, activity_type, split_type) :
    data_loader_additional = DataLoader('additional')
    additional_file_names = data_loader_additional.load_additional_data(athletes_name=athletes_name,
                                                                        activity_type=activity_type,
                                                                        split_type=split_type)
    empty_files = []
    for file_name in additional_file_names :
        print('\nCleaning {} ...'.format(file_name[3 :]))
        athlete_df = pd.DataFrame(pd.read_csv(file_name))
        addtional_data_cleaner = AdditionalDataCleaner(athlete_df, file_name)
        if addtional_data_cleaner.check_empty() :
            print('File is empty.')
            empty_files.append(file_name)
        else :
            addtional_data_cleaner.process_data_cleaning()
            cleaned_df = addtional_data_cleaner.dataframe
            _save_cleaned_df('additional', athletes_name, file_name, cleaned_df)
            missing_val_log = pd.DataFrame(addtional_data_cleaner.missing_val_logger,
                                           columns=['Before/After Imputation'])
            _save_log(data_type='additional', log_type='missing_value', file_name=file_name,
                      athletes_name=athletes_name, log_df=missing_val_log)
            outlier_log_df = pd.DataFrame(addtional_data_cleaner.outlier_dict_logger)
            _save_log(data_type='additional', log_type='outlier', file_name=file_name,
                      athletes_name=athletes_name, log_df=outlier_log_df)
    print('\nFor {}\'s additional data, {} out of {} {} files are empty.'.format(athletes_name,
                                                                                 len(empty_files),
                                                                                 len(additional_file_names),
                                                                                 activity_type))


def main(data_type='original', athletes_name: str = None, activity_type: str = None, split_type: str = None) :
    """The main function of processing data cleaning

    Parameters
    -------
    data_type: str
       The type of the data, original or additional.
    athletes_name: str
        The name of the athlete whose data is about to clean.
    activity_type: str
        The activity type of the data. Only available for additional datasets.
        Eg. 'cycling', 'running', 'swimming'.
    split_type: str
        The split type of the fit files. Only available for additional datasets.
        Eg. 'laps', 'real-time', 'starts'.
    """
    _create_cleaned_data_folder(data_type)
    _create_log_folders(data_type, athletes_name)

    if data_type == 'original' :
        if athletes_name is None :
            # Clean all original data
            orig_file_names = utility.get_all_original_data_file_names()
            for file_name in orig_file_names :
                _main_helper_original(file_name=file_name)
        else :
            # Clean data for the given athlete
            _main_helper_original(athletes_name=athletes_name)

    elif data_type == 'additional' :
        # Clean all additional data for the given athlete
        _main_helper_additional(athletes_name, activity_type, split_type)


if __name__ == '__main__':
    athletes_names = ['eduardo oliveira']

    # Clean original data
    # main('original')  # clean all original data
    main('original', athletes_names[0])  # clean original data for one athlete

    # # Clean additional data
    # activity_type = ['cycling', 'running', 'swimming']
    # split_type = 'real-time'
    # main('additional', athletes_name=athletes_names[0], activity_type=activity_type[0], split_type=split_type)
