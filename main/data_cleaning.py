"""Data Cleaning

This script allows the user to process data cleaning for both CoachingMate data and Garmin data.

This script requires that `pandas`, `numpy` be installed within the Python
environment you are running this script in.

This file can also be imported as a module
"""

import pandas as pd


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

    def __init__(self, file_name):
        self.file_name = file_name

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

    def process_data_cleaning(self):
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

    def __init__(self, file_names):
        self.file_names = file_names

    def process_data_cleaning(self):
        """Process the data cleaning

        Returns
        -------
        cleaned_df : pandas data frame
            Cleaned athlete Garmin data
        """
        pass

    def clean_numerical_columns(self, df, columns):
        pass

    def clean_categorical_columns(self, df, columns):
        pass


if __name__ == '__main__':
    addtional_data_cleaner = AdditionalDataCleaner()
