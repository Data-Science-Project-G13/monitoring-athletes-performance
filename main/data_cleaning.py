"""Data Cleaning

This script allows the user to process data cleaning for both CoachingMate data and Garmin data.

This script requires that `pandas`, `numpy` be installed within the Python
environment you are running this script in.

This file can also be imported as a module
"""

import pandas as pd


class CoachingMateDataCleaner():
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


class GarminDataCleaner():
    """
    A class used to process data cleaning on a group of Garmin datasets for one athlete

    ...

    Attributes
    ----------
    dir_name : str
      The name of the directory at where Garmin data for a certain athlete located

    Methods
    -------
    process_data_cleaning()
       Process the data cleaning
    """

    def __init__(self, dir_name):
        self.dir_name = dir_name

    def process_data_cleaning(self):
        """Process the data cleaning

        Returns
        -------
        cleaned_df : pandas data frame
            Cleaned athlete Garmin data
        """
        pass


if __name__ == '__main__':
    pass