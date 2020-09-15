"""Data Cleaning

This script allows the user to process data cleaning for both CoachingMate data and Garmin data.

This script requires that `pandas`, `numpy` be installed within the Python
environment you are running this script in.

This file can also be imported as a module
"""
# TODO: Clean up packages
# Packages
import numpy as np
import pandas as pd
import os
import datetime as dt
import pandas_profiling
import warnings
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import missingno as msno
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
# Self-defined modules
import utility
from data_loader import DataLoader
# Settings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (12, 8)
pd.options.mode.chained_assignment = None
sns.set(style="darkgrid", palette="pastel", color_codes=True)
sns.set_context('talk')
# Set the data frame display option
pd.set_option('display.max_row', 20)
pd.set_option('display.max_columns', 10)


class SpreadsheetDataCleaner():
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
        self.dataframe = dataframe
        self.dataframe_work_on = self.dataframe.copy()
        self.columns_to_numeric = utility.get_numerical_columns('spreadsheet')
        self.column_groups_imputation = utility.get_column_groups_for_imputation('spreadsheet')
        self.missing_val_logger = []
        self.outlier_dict_logger = {'Outlier Index': [], 'Column': [], 'Reason': []}

    def _split_dataframe_by_activity(self):
        dataframe_swim = self.dataframe_work_on.loc[
            self.dataframe_work_on['Activity Type'].isin(['Pool Swimming', 'Open Water Swimming', 'Swimming'])]
        dataframe_cycle = self.dataframe_work_on.loc[
            self.dataframe_work_on['Activity Type'].isin(['Virtual Cycling', 'Indoor Cycling', 'Road Cycling', 'Cycling'])]
        dataframe_run = self.dataframe_work_on.loc[
            self.dataframe_work_on['Activity Type'].isin(['Running', 'Treadmill Running'])]
        dataframe_st = self.dataframe_work_on.loc[
            self.dataframe_work_on['Activity Type'].isin(['Strength Training'])]
        dataframe_others = self.dataframe_work_on.loc[
            self.dataframe_work_on['Activity Type'].isin(['Hiking', 'Multisport', 'Indoor Rowing'])]
        return dataframe_swim, dataframe_cycle, dataframe_run, dataframe_st, dataframe_others

    # def _concat_dataframe_by_activity(self):
    #     self.dataframe_work_on = pd.concat(
    #         [self.dataframe_swim, self.dataframe_cycle, self.dataframe_run,self.dataframe_st, self.dataframe_others])
    #     del self.dataframe_swim, self.dataframe_cycle, self.dataframe_run,self.dataframe_st, self.dataframe_others
    #     self.dataframe_work_on = self.dataframe_work_on.sort_index(inplace=True)

    def _drop_columns(self) :
        columns_to_drop = [ #'Avg HR', 'Max HR', 'Training Stress Score®',
                           'Favorite', 'Aerobic TE', 'Avg Run Cadence', 'Max Run Cadence', 'Avg Stride Length',
                           'Avg Vertical Ratio', 'Avg Vertical Oscillation', 'Avg Ground Contact Time',
                           'Avg GCT Balance', 'L/R Balance', 'Grit', 'Flow', 'Total Reps', 'Total Sets',
                           'Bottom Time', 'Min Temp', 'Surface Interval', 'Decompression', 'Best Lap Time', 'Max Temp']
        self.dataframe_work_on.drop(columns_to_drop, axis=1, inplace=True)

    def _convert_strings_to_lower_case(self) :
        self.dataframe_work_on['Activity Type'] = self.dataframe_work_on['Activity Type'].str.lower()
        self.dataframe_work_on['Title'] = self.dataframe_work_on['Title'].str.lower()

    def _handle_commas(self) :
        columns_remove_comma = self.dataframe_work_on.columns
        # TODO: Might not simply replace the comma with an empty string
        for column in columns_remove_comma :
            self.dataframe_work_on[column] = self.dataframe_work_on[column].astype(str).str.replace(',', '')

    def _format_missing_val_with_nan(self) :
        # TODO: Missing value situations in config and in functions to handle
        self.dataframe_work_on = self.dataframe_work_on.replace({"--" : np.nan, "..." : np.nan})
        self.dataframe_work_on.loc[self.dataframe_work_on['Max Speed'].str.contains(":", na=False), 'Max Speed'] = np.nan
        self.dataframe_work_on.loc[self.dataframe_work_on['Avg Speed'].str.contains(":", na=False), 'Avg Speed'] = np.nan


    def _convert_columns_to_numeric(self) :
        self.dataframe_work_on[self.columns_to_numeric] = self.dataframe_work_on[self.columns_to_numeric].apply(pd.to_numeric)


    def _format_datetime(self):
        self.dataframe_work_on['Date_extracted'] = pd.to_datetime(self.dataframe_work_on["Date"]).dt.normalize()
        self.dataframe_work_on['Time_extracted'] = pd.to_datetime(self.dataframe_work_on["Date"]).dt.time
        self.dataframe_work_on['Date'] = pd.to_datetime(self.dataframe_work_on['Date'])
        self.dataframe_work_on['Time_sec'] = pd.to_timedelta(
            pd.to_datetime(self.dataframe_work_on["Time"]).dt.strftime('%H:%M:%S')).dt.total_seconds()

    def _find_missing_percent(self, dataframe):
        """
        Returns dataframe containing the total missing values and percentage of total
        missing values of a column.
        """
        missing_val_df, colNames = pd.DataFrame({'ColumnName' : [], 'TotalMissingVals' : [], 'PercentMissing' : []}), []
        for col in dataframe.columns :
            sum_miss_val = dataframe[col].isnull().sum()
            percent_miss_val = round((sum_miss_val / dataframe.shape[0]) * 100, 2)
            missing_val_df = missing_val_df.append(
                dict(zip(missing_val_df.columns, [col, sum_miss_val, percent_miss_val])),
                ignore_index=True)
            missing_above_80 = missing_val_df[missing_val_df['PercentMissing'] > 60.0]
            colNames = missing_above_80['ColumnName']
            colNames = colNames.tolist()
        dataframe = dataframe.drop(colNames, axis=1)
        print("Columns with at least 80% missing values", colNames)
        return colNames

    def plot_missing_val_bar(self) :
        graph = msno.bar(self.dataframe_work_on)
        return graph

    def get_missingno_matrix(self) :
        matrix = msno.matrix(self.dataframe_work_on)
        return matrix

    def make_heatmap(self) :
        heatmap = msno.heatmap(self.dataframe_work_on)
        return heatmap

    def get_deno_gram(self) :
        dendogram = msno.dendrogram(self.dataframe_work_on)
        return (dendogram)

    def get_profile_report(self) :
        return pandas_profiling.ProfileReport(self.dataframe_work_on)

    def get_numerical_columns(self) :
        numeric_column_df = self.dataframe_work_on.select_dtypes(include=[np.number])
        numeric_column_columns = numeric_column_df.columns.values
        return numeric_column_columns

    def get_categorical_columns(self) :
        categorical_df = self.dataframe_work_on.select_dtypes(exclude=[np.number])
        categorical_columns = categorical_df.columns.values
        return categorical_columns

    def _apply_mean_imputation(self, sub_df_work_on, columns):
        not_null_cols = [col for col in columns if col in sub_df_work_on.columns]
        not_null_cols = [col for col in not_null_cols if not sub_df_work_on[col].isnull().all()]
        # print(sub_df_work_on[not_null_cols])
        for column in not_null_cols:
            sub_df_work_on[column].fillna(sub_df_work_on[column].mean(), inplace=True)
        # new_data = sub_df_work_on.copy()
        # imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        # new_data = pd.DataFrame(imputer.fit_transform(new_data[not_null_cols]), columns=[not_null_cols])
        # sub_df_work_on[not_null_cols] = new_data[not_null_cols]
        # print(sub_df_work_on[not_null_cols])

    def _apply_mice_imputation(self, sub_df_work_on, columns):
        not_null_cols = [col for col in columns if not sub_df_work_on[col].isnull().all()]
        new_data = sub_df_work_on.copy()
        imputer = IterativeImputer(GradientBoostingRegressor())
        new_data = pd.DataFrame(imputer.fit_transform(new_data[not_null_cols]), columns=[not_null_cols])
        sub_df_work_on[not_null_cols] = new_data[not_null_cols]

    def _apply_knn_imputation(self, sub_df_work_on, columns):
        not_null_cols = [col for col in columns if not sub_df_work_on[col].isnull().all()]
        new_data = sub_df_work_on.copy()
        imputer = KNNImputer(n_neighbors=23)
        new_data = pd.DataFrame(imputer.fit_transform(new_data[not_null_cols]), columns=[not_null_cols])
        sub_df_work_on[not_null_cols] = new_data[not_null_cols]

    def _apply_mode_imputation(self, dataframe_work_on_categorical):
        """ Mode Imputation
        """
        for col in dataframe_work_on_categorical.columns:
            mode = dataframe_work_on_categorical[col].mode().iloc[0]
            dataframe_work_on_categorical[col] = dataframe_work_on_categorical[col].fillna(mode)

    def _apply_imputations(self, sub_df_work_on, columns_need_imputation, column_type="numerical", impute_tech="mean"):
        if column_type == 'numerical':
            if impute_tech == "mean":
                self._apply_mean_imputation(sub_df_work_on, columns_need_imputation)
            elif impute_tech == "mice":
                self._apply_mice_imputation(sub_df_work_on, columns_need_imputation)
            elif impute_tech == "knn":
                self._apply_knn_imputation(sub_df_work_on, columns_need_imputation)
        else:
            self._apply_mode_imputation(sub_df_work_on)

    def _apply_linear_interpolation(self, numeric_column_df):
        for col in numeric_column_df.columns:
            numeric = numeric_column_df.interpolate(method='linear', limit_direction='forward', axis=0).ffill().bfill()

    def get_minmax(self, numeric_column_df, numeric_column_values):
        # when imputing a knn data must be normalised to reduce the bias in the imputation
        scaler = MinMaxScaler()
        scaling = pd.DataFrame(scaler.fit_transform(numeric_column_df), columns=numeric_column_values)
        return scaling

    def _find_missing_index(self, data_numeric_regr, target_cols) :
        miss_index_dict = {}
        for tcol in target_cols :
            index = data_numeric_regr[tcol][data_numeric_regr[tcol].isnull()].index
            miss_index_dict[tcol] = index
        return miss_index_dict

    # def _apply_regression_imputation(self, data_numeric_regr, target_cols, miss_index_dict) :
    #     '''Predictors for regression imputation'''
    #     predictors = data_numeric_regr.drop(target_cols, axis=1)
    #     for tcol in target_cols :
    #         y = data_numeric_regr[tcol]
    #         '''Initially impute the column with mean'''
    #         y = y.fillna(y.mean())
    #         xgb = xgboost.XGBRegressor(objective="reg:squarederror", random_state=42)
    #         '''Fit the model where y is the target column which is to be imputed'''
    #         xgb.fit(predictors, y)
    #         predictions = pd.Series(xgb.predict(predictors), index=y.index)
    #         index = miss_index_dict[tcol]
    #         '''Replace the missing values with the predictions'''
    #         data_numeric_regr[tcol].loc[index] = predictions.loc[index]
    #     return data_numeric_regr

    def out_iqr(self,numeric_column_values):
        for index in numeric_column_values:
            q25, q75 = np.quantile(self.dataframe_work_on[index], 0.25), np.quantile(self.dataframe_work_on[index], 0.75)
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
            df1 = self.dataframe_work_on[self.dataframe_work_on[index] > upper]
            df2 = self.dataframe_work_on[self.dataframe_work_on[index] < lower]
            print("Total number of outliers for the column ",index," are", df1.shape[0]+ df2.shape[0])
            df = pd.concat([df1, df2])
            print("Outlier indexes are", df.index)
            del df, df1, df2

    def out_plot(self, column):
        plt.figure(figsize=(10, 6))
        sns.distplot(self.dataframe_work_on[column], kde=False)
        q25, q75 = np.quantile(self.dataframe_work_on[column], 0.25), np.quantile(self.dataframe_work_on[column], 0.75)
        # calculate the IQR
        iqr = q75 - q25
        # calculate the outlier cutoff
        cut_off = iqr * 1.5
        # calculate the lower and upper bound value
        lower, upper = q25 - cut_off, q75 + cut_off
        plt.axvspan(xmin=lower, xmax=self.dataframe_work_on[column].min(), alpha=0.2, color='red')
        plt.axvspan(xmin=upper, xmax=self.dataframe_work_on[column].max(), alpha=0.2, color='red')
        plt.show()
        sns.distplot(self.dataframe_work_on[column])
        plt.show()

    def out_std(self,numeric_column_values) :
        for index in numeric_column_values:
            # calculate the mean and standard deviation of the data frame
            data_mean, data_std = self.dataframe_work_on[index].mean(), self.dataframe_work_on[index].std()
            # calculate the cutoff value
            cut_off = data_std * 3
            # calculate the lower and upper bound value
            lower, upper = data_mean - cut_off, data_mean + cut_off
            print('The lower bound value is', lower)
            print('The upper bound value is', upper)
            # Calculate the number of records below and above lower and above bound value respectively
            df1 = self.dataframe_work_on[index][self.dataframe_work_on[index] > upper]
            df2 = self.dataframe_work_on[index][self.dataframe_work_on[index] < lower]
            print("Total number of outliers for the column ",index," are", df1.shape[0]+ df2.shape[0])
            df=pd.concat([df1,df2])
            print("Outlier indexes are",df.index)
            del df,df1,df2

    def out_zscore(self,numeric_column_values) :
        threshold = 3
        for index in numeric_column_values:
            outliers = []
            zscore = []
            indices=[]
            mean = np.mean(self.dataframe_work_on[index])
            std = np.std(self.dataframe_work_on[index])
            for i,j in zip(self.dataframe_work_on[index], self.dataframe_work_on[index].index):
                z_score = (i - mean) / std
                zscore.append(z_score)
                if np.abs(z_score) > threshold :
                    outliers.append(i)
                    indices.append(j)
            print("Total number of outliers for the column ",index, "are", len(outliers),"and its rowindexes are",indices)

    def localOutlierFactor(self,numeric_column_values):
        clf = LocalOutlierFactor(n_neighbors=50, contamination='auto')
        X = self.dataframe_work_on[numeric_column_values].values
        y_pred = clf.fit_predict(X)
        indices=np.where(y_pred == -1)
        y_pred = np.unique(y_pred, return_counts=True)
        return print("Total number of outliers are denoted by -1",y_pred,"The outliers are at indices",indices)

    def process_data_cleaning(self) :
        """
        Process data cleaning for spreadsheet data
        Returns
        -------
        cleaned_df : pandas DataFrame
            Cleaned athlete spreadsheet data
        """
        # ================ Base Clean ====================
        self._drop_columns()
        # self._convert_strings_to_lower_case()
        self._handle_commas()
        self._format_missing_val_with_nan()
        self._convert_columns_to_numeric()
        self._format_datetime()
        self._convert_columns_to_numeric()
        numerical_columns = self.get_numerical_columns()
        categorical_columns = self.get_categorical_columns()

        for sub_df_work_on in self._split_dataframe_by_activity():
            # ================ Imputations ====================
            print('=======================')
            columns_to_drop = self._find_missing_percent(sub_df_work_on)
            columns_keep = [column for column in sub_df_work_on.columns if column not in columns_to_drop]
            sub_df_work_on = sub_df_work_on[columns_keep]
            if not sub_df_work_on.empty:
                self._apply_imputations(sub_df_work_on, numerical_columns)
        self.dataframe[self.dataframe_work_on.columns] = self.dataframe_work_on[self.dataframe_work_on.columns]

        # print(list(self.dataframe.isna().any()))

        # data_numeric = self.dataframe[numeric_column_values]
        # self._apply_mean_imputation(data_numeric)
        # data_numeric_regr = self.dataframe[numeric_column_values]
        # # '''Numeric columns with missing values which acts as target in training'''
        # target_cols = ['Avg. Swolf', 'Total Strokes', 'Avg Speed', 'Avg HR', 'Max HR', 'Avg Bike Cadence',
        #                'Max Bike Cadence', "Distance", "Calories", "Avg Stroke Rate", "Number of Laps"]
        # data_numeric_regr.drop(target_cols, axis=1)
        # miss_index_dict = self._find_missing_index(data_numeric_regr, target_cols)
        # self._apply_regression_imputation(data_numeric_regr, target_cols, miss_index_dict)
        # self._apply_linear_interpolation(numeric_column_df)
        # self.get_minmax(numeric_column_df, numeric_column_values)
        # self._apply_knn_imputation(numeric_column_df, numeric_column_values)  # assigning imputaion can change later

        # ================ Outliers ====================
        # self.out_iqr(numerical_columns)
        # self.out_std(numerical_columns)
        # self.out_zscore(numerical_columns)
        # # self.out_plot("Distance")
        # self.localOutlierFactor(numerical_columns)


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

    def __init__(self, dataframe: pd.DataFrame, file_name: str = None):
        self.dataframe = dataframe
        self.file_name = file_name
        self.column_groups_imputation = utility.get_column_groups_for_imputation('additional')
        self.numerical_ordered_columns = utility.get_numerical_columns('additional', 'ordered')
        self.numerical_fluctuating_columns = utility.get_numerical_columns('additional', 'fluctuating')
        self.categorical_columns = utility.get_categorical_columns('additional')
        self.missing_val_logger = []
        self.outlier_dict_logger = {'Outlier Index' : [], 'Column' : [], 'Reason' : []}

        self.color_labels = utility.get_outlier_color_labels_additional()
        self.OUTLIER_COLOR_LABEL, self.TIME_DIFF_THRESHOLD, self.ROW_COUNT_THRESHOLD = len(self.color_labels)-1, 3 * 60, 150  # Configurable
        self.num_records, self.time_sgmt_num = -1, 0


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
        no_missing = True
        missing_val_perc = self.dataframe.isnull().sum() / self.dataframe.shape[0]
        for column_name in missing_val_perc.keys():
            if float(missing_val_perc[column_name]) != 0:
                no_missing = False
                self.missing_val_logger.append(
                    '{} has {}% missing. \n'.format(column_name, round(missing_val_perc[column_name] * 100, 3)))
        if no_missing:
            self.missing_val_logger.append('Great! No missing values! \n')
        return missing_val_perc

    def get_columns_with_missing_values(self):
        cols_with_missing = {col for col in self.dataframe.columns if self.dataframe[col].isnull().any()}
        return cols_with_missing

    def convert_str_to_num_in_numerical_cols(self):
        for column in self.numerical_ordered_columns + self.numerical_fluctuating_columns:
            # dataframe[column] = dataframe[column].astype(float)
            self.dataframe[column] = pd.to_numeric(self.dataframe[column], errors='coerce')

    def add_column_time_in_seconds(self):
        time_in_seconds = []
        for time in self.dataframe['timestamp']:
            t = dt.datetime.strptime(time[:-6], '%Y-%m-%d %H:%M:%S')
            seconds = int(dt.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds())
            time_in_seconds.append(seconds)
        self.dataframe.insert(1, 'time_in_seconds', time_in_seconds, True)

    def _apply_univariate_imputation(self, columns):
        if columns:
            new_data = self.dataframe.copy()
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            new_data = pd.DataFrame(imputer.fit_transform(new_data[columns]), columns = columns)
            self.dataframe[columns] = new_data[columns]
            del new_data

    def _apply_multivariate_imputation(self, columns_need_imputation, column_group):
        null_cols = [col for col in columns_need_imputation if self.dataframe[col].isnull().all()]
        if len(null_cols) == len(column_group):
            self.missing_val_logger.append("All the values in column(s) {} are missing. "
                                           "Not able to apply imputation.".format(null_cols))
        elif len(null_cols) > 0:
            cols_not_null = [col for col in columns_need_imputation if col not in null_cols]
            self._apply_univariate_imputation(cols_not_null)
            self._apply_regression_prediction_imputation(null_cols)
        else:
            new_data = self.dataframe.copy()
            try:
                new_data.replace('None', None)
                iter_imputer = IterativeImputer(max_iter=10, random_state=0)
                new_data = pd.DataFrame(iter_imputer.fit_transform(new_data[column_group]))
                if not new_data.empty :
                    new_data.columns = column_group
                    self.dataframe[column_group] = new_data[column_group]
                else:
                    self.missing_val_logger.append("All the values in column(s) {} are missing. "
                                                   "Not able to apply imputation.".format(column_group))
            except:
                pass
            del new_data


    def _apply_interpolation_imputation(self, columns):
        for column in columns :
            interpolated_column = self.dataframe[column].interpolate(method='spline', order=2, limit=2)
            self.dataframe[column] = interpolated_column

    def _apply_regression_prediction_imputation(self, column_names):
        # TODO: for whole column missing
        pass

    def _apply_imputations(self, columns_need_imputation):
        for impute_tech, column_names in self.column_groups_imputation.items():
            column_intersection = column_names.intersection(columns_need_imputation)
            if column_intersection:
                if impute_tech == "univariate":
                    self._apply_univariate_imputation(list(column_intersection))
                elif impute_tech == "multivariate1" or "multivariate2":
                    if len(column_intersection) > 0:
                        self._apply_multivariate_imputation(list(column_intersection), list(column_names))
                elif impute_tech == "interpolation":
                    self._apply_interpolation_imputation(list(column_intersection))

    def handle_missing_values(self):
        self.format_missing_val_with_nan()
        self.missing_val_logger.append('Before imputation: \n')
        self.check_missing_val_perc()
        columns_need_imputation = self.get_columns_with_missing_values()
        if columns_need_imputation:
            self._apply_imputations(columns_need_imputation)
        self.missing_val_logger.append('After imputation: \n')
        self.check_missing_val_perc()

    def _log_outlier(self, outlier_index, column, reason):
        self.outlier_dict_logger['Outlier Index'].append(outlier_index)
        self.outlier_dict_logger['Column'].append(column)
        self.outlier_dict_logger['Reason'].append(reason)

    def _filter_by_repetition(self, rec_color):
        df = self.dataframe.drop(columns=['timestamp', 'timestamp_utc', 'timezone'])
        dup_df = df[df.duplicated()]
        idx_list = dup_df.index.tolist()
        for i in idx_list:
            rec_color[i] = self.OUTLIER_COLOR_LABEL
            self._log_outlier(i, 'ALL', 'DUPLICATE ROW')

    def _filter_by_timestamp(self, rec_color):
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

    def _filter_by_continuity_assumption(self, i, rec_color):
        # The change of temperature should not exceed 1 degree
        if (rec_color[i] == rec_color[i - 1] and rec_color[i] != self.OUTLIER_COLOR_LABEL):
            diff = abs(self.dataframe.loc[i, 'temperature'] - self.dataframe.loc[i - 1, 'temperature'])
            if diff > 1:
                rec_color[i] = self.OUTLIER_COLOR_LABEL
                self._log_outlier(i, 'temperature', 'TEMPERATURE CHANGES TOO FAST')
                # print("[OUTLIER DETECTED!] Index == %d Reason: TEMPERATURE CHANGES TOO FAST!" % (i))

    def _filter_by_logical_inconsistency(self, i, rec_color):
        # distance, enhanced_speed, speed, heart_rate and cadence must be greater than or equal to 0
        for column in ['distance', 'enhanced_speed', 'speed', 'heart_rate', 'cadence']:
            if self.dataframe[column][i] < 0:
                # take the zero values as missing values
                rec_color[i] = self.OUTLIER_COLOR_LABEL
                self._log_outlier(i, column, 'LOGICAL ERROR, {} has to be non-negative!'.format(column))

    def _filter_by_zscore(self, dataframe, rec_color):
        # Configuraing different tolerace for different columns
        columns = ['position_lat', 'position_long', 'enhanced_altitude', 'altitude', 'heart_rate', 'cadence']
        tolerances = [2, 2, 3, 3, 3, 3]  # Configurable
        # TODO: Avoid adding column 'color' to the dataframe
        dataframe['color'] = rec_color
        for k in range(self.time_sgmt_num):
            df = dataframe.loc[dataframe['color'] == k]
            for j in range(len(columns)):
                # Actual Z-score test
                z = np.abs(stats.zscore(df[columns[j]]))
                outlier_rows = np.where(z > tolerances[j])
                # print("\n====For Column:%s, There are %d outliers ===\n" % (columns[j], len(outlier_rows[0])))
                for i in outlier_rows[0]:
                    rec_color[i] = self.OUTLIER_COLOR_LABEL
                    self._log_outlier(i, columns[j],
                                      'Z-SCORE({}) EXCEEDS THE TOLERANCE ({})'.format(z[i], tolerances[j]))

    def _boxplot(self, df, rec_color):
        """ Boxplot for the outliers
        :param df:
        :param rec_color:
        :return:
        """
        plt.boxplot(df, sym="o", whis=1.5)
        plt.show()
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        IQR_predicate = (df < Q1 - 1.5 * IQR) | (df > Q3 + 1.5 * IQR)
        dropped = df[IQR_predicate]
        for idx in dropped :
            rec_color[idx] = self.OUTLIER_COLOR_LABEL

    def _plot_time_sgmt(self, colors):
        ts_secs = np.array(self.dataframe['time_in_seconds'])
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
        outlier_mask = [1 if rec_color[i] == self.OUTLIER_COLOR_LABEL else 0 for i in range(self.num_records)]
        return outlier_mask

    def get_outliers_by_zscore(self):
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

    def get_outliers_timestamp(self):
        pass

    def handle_outliers_numerical_ordered(self):
        pass

    def handle_outliers_numerical_fluctuating(self):
        pass

    def handle_outliers(self):
        """
        return: (outlier_mask, df_outlier_free)
            outlier_mask: 0 <==> not outlier; 1 <==> outlier
            df_outlier_free: dataframe without possible outliers
        """
        self.convert_str_to_num_in_numerical_cols()
        dataframe = self.dataframe.copy()
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

        for i in range(1, self.num_records):
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
        return outlier_mask, df_outlier_free

    def process_data_cleaning(self):
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


def _create_cleaned_data_folder(data_type):
    if data_type == 'spreadsheet':
        cleaned_spreadsheet_folder = '{}/data/cleaned_spreadsheet'.format(os.path.pardir)
        if not os.path.exists(cleaned_spreadsheet_folder):
            os.mkdir(cleaned_spreadsheet_folder)
    elif data_type == 'additional':
        cleaned_additional_folder = '{}/data/cleaned_additional'.format(os.path.pardir)
        if not os.path.exists(cleaned_additional_folder):
            os.mkdir(cleaned_additional_folder)
    else:
        raise Exception('No {} type of datasets'.format(data_type))


def _create_log_folders(data_type, athletes_name=None):
    if not os.path.exists('{}/log/'.format(os.path.pardir)):
        os.mkdir('{}/log/'.format(os.path.pardir))

    if data_type == 'spreadsheet':
        log_folder_names = ['spreadsheet_missing_value_log', 'spreadsheet_outlier_log']
    elif data_type == 'additional':
        log_folder_names = ['additional_missing_value_log', 'additional_outlier_log']
        if athletes_name:
            log_folder_names.extend(['{}/{}'.format(name, athletes_name) for name in log_folder_names])
    else:
        raise Exception('No {} type of datasets'.format(data_type))

    for log_folder_name in log_folder_names:
        folder = '{}/log/{}'.format(os.path.pardir, log_folder_name)
        if not os.path.exists(folder):
            os.mkdir(folder)


def _save_cleaned_df(data_type, athletes_name, file_name, cleaned_df, verbose=False):
    if data_type == 'spreadsheet':
        cleaned_df.to_csv('{}/data/cleaned_spreadsheet/{}'.format(os.path.pardir, file_name), index=False)
        if verbose: print('Cleaned {} data saved.'.format(file_name))

    elif data_type == 'additional':
        athlete_cleaned_additional_data_folder = '{}/data/cleaned_additional/{}'.format(os.path.pardir,
                                                                                        '_'.join(athletes_name.lower().split()))
        if not os.path.exists(athlete_cleaned_additional_data_folder):
            os.mkdir(athlete_cleaned_additional_data_folder)
        cleaned_df.to_csv('{}/{}'.format(athlete_cleaned_additional_data_folder, file_name.split('/')[-1]), index=False)
        if verbose: print('{}\'s cleaned {} data saved.'.format(athletes_name.capitalize(), file_name.split('/')[-1]))


def _save_log(data_type, log_type, file_name, log_df, athletes_name=None):
    if data_type == 'spreadsheet':
        log_file_path = '{}/log/{}_{}_log/{}'.format(os.path.pardir, data_type, log_type, file_name.split('/')[-1])
    elif data_type == 'additional':
        log_file_path = '{}/log/{}_{}_log/{}/{}'.format(os.path.pardir, data_type, log_type, athletes_name,
                                                        file_name.split('/')[-1])
    else:
        raise Exception('No {} type of datasets'.format(data_type))
    if not log_df.empty:
        log_df.to_csv(log_file_path, index=False)


def _main_helper_spreadsheet(athletes_name=None, file_name: str = None, verbose=False):
    # TODO: Create missing value and outlier log for spreadsheet data
    data_loader_spreadsheet = DataLoader('spreadsheet')
    if file_name :
        athlete_df = data_loader_spreadsheet.load_spreadsheet_data(file_name=file_name)
    else :
        athlete_df = data_loader_spreadsheet.load_spreadsheet_data(athletes_name=athletes_name)
        file_name = data_loader_spreadsheet.config.get('SPREADSHEET-DATA-SETS', athletes_name.lower())
    spreadsheet_data_cleaner = SpreadsheetDataCleaner(athlete_df)
    spreadsheet_data_cleaner.process_data_cleaning()
    cleaned_df = spreadsheet_data_cleaner.dataframe_work_on
    _save_cleaned_df('spreadsheet', athletes_name, file_name, cleaned_df, verbose=verbose)


def _main_helper_additional(athletes_name, activity_type, split_type, verbose=False):
    data_loader_additional = DataLoader('additional')
    additional_file_names = data_loader_additional.load_additional_data(athletes_name=athletes_name,
                                                                        activity_type=activity_type,
                                                                        split_type=split_type)
    empty_files = []
    for file_name in additional_file_names:
        if verbose: print('\nCleaning {} ...'.format(file_name[3:]))
        athlete_df = pd.DataFrame(pd.read_csv(file_name))
        addtional_data_cleaner = AdditionalDataCleaner(athlete_df, file_name)
        if addtional_data_cleaner.check_empty():
            if verbose: print('File is empty.')
            empty_files.append(file_name)
        else:
            addtional_data_cleaner.process_data_cleaning()
            cleaned_df = addtional_data_cleaner.dataframe
            _save_cleaned_df('additional', athletes_name, file_name, cleaned_df, verbose=verbose)
            missing_val_log = pd.DataFrame(addtional_data_cleaner.missing_val_logger,
                                           columns=['Before/After Imputation'])
            _save_log(data_type='additional', log_type='missing_value', file_name=file_name,
                      athletes_name=athletes_name, log_df=missing_val_log)
            outlier_log_df = pd.DataFrame(addtional_data_cleaner.outlier_dict_logger)
            _save_log(data_type='additional', log_type='outlier', file_name=file_name,
                      athletes_name=athletes_name, log_df=outlier_log_df)
    if verbose: print('\nFor {}\'s additional data, {} out of {} {} files are empty.'.format(athletes_name,
                                                                                 len(empty_files),
                                                                                 len(additional_file_names),
                                                                                 activity_type))


def main(data_type='spreadsheet', athletes_name: str = None, activity_type: str = None, split_type: str = None):
    """The main function of processing data cleaning

    Parameters
    -------
    data_type: str
       The type of the data, spreadsheet or additional.
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

    if data_type == 'spreadsheet':
        if athletes_name is None:
            # Clean all spreadsheet data
            spreadsheet_file_names = utility.get_all_spreadsheet_data_file_names()
            for file_name in spreadsheet_file_names:
                _main_helper_spreadsheet(file_name=file_name, verbose=True)
        else:
            # Clean data for the given athlete
            _main_helper_spreadsheet(athletes_name=athletes_name, verbose=True)

    elif data_type == 'additional':
        # Clean all additional data for the given athlete
        _main_helper_additional(athletes_name, activity_type, split_type, verbose=True)


if __name__ == '__main__':
    athletes_names = ['eduardo oliveira']

    # Clean spreadsheet data
    # main('spreadsheet')  # clean all spreadsheet data
    main('spreadsheet', athletes_name=athletes_names[0])  # clean spreadsheet data for one athlete

    # Clean additional data
    activity_types = ['cycling', 'running', 'swimming']
    split_type = 'real-time'
    for activity_type in activity_types:
        main('additional', athletes_name=athletes_names[0], activity_type=activity_type, split_type=split_type)
