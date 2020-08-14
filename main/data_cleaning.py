"""Data Cleaning

This script allows the user to process data cleaning for both CoachingMate data and Garmin data.

This script requires that `pandas`, `numpy` be installed within the Python
environment you are running this script in.

This file can also be imported as a module
"""

import pandas as pd
from data_loader import DataLoader

# Set the data frame display option
pd.set_option('display.max_row', 10)
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
        pass

    def show_missing_val_distribution(self, dataframe):
        pass

    def show_missing_val_locations(self, dataframe):
        pass

    def make_heatmap(self, dataframe):
        pass

    def make_correlation_plot(self, dataframe):
        pass

    def replace_missing_vals_with_nan(self, dataframe):
        pass

    def formalize_dates(self, dataframe):
        pass

    def process_data_cleaning(self, dataframe):
        """Process the data cleaning

        Returns
        -------
        cleaned_df : pandas data frame
            Cleaned athlete CoachingMate data
        """
        pass


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

    def __init__(self):
        pass
        # self.file_names = file_names

    def check_empty(self, dataframe):
        if dataframe.empty: return True
        else: return False

    def check_missing_val_perc(self, dataframe: pd.DataFrame):
        missing_val_counts = dataframe.isnull().sum()/dataframe.shape[0]
        print(missing_val_counts)

    def check_outliers(self, dataframe, columns_focus_on=None):
        pass

    def clean_numerical_columns(self, dataframe, columns_focus_on=None):
        pass

    def clean_categorical_columns(self, dataframe, columns_focus_on=None):
        pass

    def process_data_cleaning(self):
        """Process the data cleaning

        Returns
        -------
        cleaned_df : pandas data frame
            Cleaned athlete Garmin data
        """
        pass


if __name__ == '__main__':

    # Clean original data
    data_loader_original = DataLoader('original')
    orig_data_names = data_loader_original.get_all_original_data_file_names()
    original_data_cleaner = OriginalDataCleaner()
    for data_name in orig_data_names:
        df = data_loader_original.load_original_data(orig_data_names[0])
        original_data_cleaner.show_missing_val_distribution(df)


    # Clean additional data
    athletes_name = 'Eduardo Oliveira'
    activity_type = 'swimming'
    split_type = 'real-time'
    data_loader_additional = DataLoader('additional')
    file_names = data_loader_additional.load_additional_data(athletes_name=athletes_name,
                                                             activity_type=activity_type,
                                                             split_type=split_type)
    addtional_data_cleaner = AdditionalDataCleaner()
    empty_files = []
    for file_name in file_names:
        df = pd.DataFrame(pd.read_csv(file_name))
        if addtional_data_cleaner.check_empty(df):
            empty_files.append(file_name)
        else:
            addtional_data_cleaner.check_missing_val_perc(df)
    print('For {}\'s additional data, {} out of {} {} files are empty.'.format(athletes_name,
                                                                               len(empty_files),
                                                                               len(file_names),
                                                                               activity_type))
