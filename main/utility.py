"""Athletes Data Loader

This script contains common functions that support the project.

This script requires that `pandas` be installed.

This file can be imported as a module and contains the following
functions:

    * get_all_original_data_file_names - return a list of original data file names
    * get_all_additional_data_folder_names - return a list of additional data folder names

    * get_original_numerical - returns a list of numerical variables in original data
    * get_original_categorical - returns a list of categorical variables in original data
    * get_additional_numerical - returns a list of numerical variables in additional data
    * get_additional_categorical - returns a list of categorical variables in additional data

    * get_original_activity_types - returns a list of activity types in original data
    * get_additional_activity_types - returns a list of activity types in additional data

"""

import os
import re
from configparser import ConfigParser


config_parser = ConfigParser()
data_names_config = '{}/main/config/data_file_names.cfg'.format(os.path.pardir)
column_data_types_config = '{}/main/config/column_data_types.cfg'.format(os.path.pardir)
activity_types_config = '{}/main/config/activity_types.cfg'.format(os.path.pardir)
pattern = re.compile("^\s+|\s*,\s*|\s+$")


def get_all_spreadsheet_data_file_names():
    """Get all the original data file names

    Returns
    -------
        List of strings
    """
    config_parser.read(data_names_config)
    return [config_parser.get('SPREADSHEET-DATA-SETS', key) for key in list(config_parser['SPREADSHEET-DATA-SETS'].keys())]


def get_all_additional_data_folder_names():
    """Get all the additional data folder names

    Returns
    -------
        List of strings
    """
    config_parser.read(data_names_config)
    return [config_parser.get('ADDITIONAL-DATA-FOLDERS', key) for key in list(config_parser['ADDITIONAL-DATA-FOLDERS'].keys())]


def get_spreadsheet_numerical():
    """Get all the numerical column names in original data

    Returns
    -------
        List of strings
    """
    config_parser.read(column_data_types_config)
    return pattern.split(config_parser.get('SPREADSHEET', 'numerical'))


def get_spreadsheet_categorical():
    """Get all the categorical column names in original data

    Returns
    -------
        List of strings
    """
    config_parser.read(column_data_types_config)
    return pattern.split(config_parser.get('SPREADSHEET', 'categorical'))


def get_all_original_data_columns():
    """Get all the column names in original data

    Returns
    -------
        columns: List of strings
            All the column names in original data
    """
    columns = get_spreadsheet_numerical() + \
              get_spreadsheet_categorical()
    return columns


def get_additional_numerical_ordered():
    """Get all the numerical column names in additional data

    Returns
    -------
        List of strings
    """
    config_parser.read(column_data_types_config)
    return pattern.split(config_parser.get('ADDITIONAL', 'numerical_ordered'))


def get_additional_numerical_fluctuating():
    """Get all the numerical column names in additional data

    Returns
    -------
        List of strings
    """
    config_parser.read(column_data_types_config)
    return pattern.split(config_parser.get('ADDITIONAL', 'numerical_fluctuating'))


def get_additional_categorical():
    """Get all the categorical column names in additional data

    Returns
    -------
        List of strings
    """
    config_parser.read(column_data_types_config)
    return pattern.split(config_parser.get('ADDITIONAL', 'categorical'))


def get_all_additional_data_columns():
    """Get all the column names in additional data

    Returns
    -------
        columns: List of strings
            All the column names in additional data
    """
    columns = get_additional_numerical_ordered() + \
              get_additional_numerical_fluctuating() + \
              get_additional_categorical()
    return columns


def get_spreadsheet_activity_types():
    """Get all the activity types in original data

    Returns
    -------
        List of strings
    """
    config_parser.read(activity_types_config)
    return pattern.split(config_parser.get('SPREADSHEET', 'activity_types'))


def get_additional_activity_types():
    """Get all the activity types in additional data

    Returns
    -------
        List of strings
    """
    config_parser.read(activity_types_config)
    return pattern.split(config_parser.get('ADDITIONAL', 'activity_types'))

def create_all_folders():
    pass


if __name__ == '__main__':
    # The lines below are for test
    print(get_all_spreadsheet_data_file_names())
    print(get_all_additional_data_folder_names())
    print(get_spreadsheet_numerical())
    print(get_spreadsheet_categorical())
    print(get_additional_numerical_ordered())
    print(get_additional_categorical())
    print(get_spreadsheet_activity_types())
    print(get_additional_activity_types())