"""Athletes Data Loader

This script contains common functions that support the project.

This script requires that `pandas` be installed.

This file can be imported as a module and contains the following
functions:
    * get_original_numerical - returns a pandas data_frame
    * get_original_categorical
    * get_additional_numerical
    * get_additional_categorical
"""

import os
from configparser import ConfigParser


config_parser = ConfigParser()
data_names_config = '{}/main/config/data_file_names.cfg'.format(os.path.pardir)
column_data_types_config = '{}/main/config/column_data_types.cfg'.format(os.path.pardir)


def get_all_original_data_file_names():
    """Get all the original data file names

    Returns
    -------
        List of strings
    """
    config_parser.read(data_names_config)
    return [config_parser.get('ORIGINAL-DATA-SETS', key) for key in list(config_parser['ORIGINAL-DATA-SETS'].keys())]


def get_all_additional_data_folder_names():
    """Get all the additional data folder names

    Returns
    -------
        List of strings
    """
    config_parser.read(data_names_config)
    return [config_parser.get('ADDITIONAL-DATA-FOLDERS', key) for key in list(config_parser['ADDITIONAL-DATA-FOLDERS'].keys())]


def get_original_numerical():
    """Get all the numerical column names in original data

    Returns
    -------
        List of strings
    """
    config_parser.read(column_data_types_config)
    return config_parser.get('ORIGINAL', 'numerical').split(',')


def get_original_categorical():
    """Get all the categorical column names in original data

    Returns
    -------
        List of strings
    """
    config_parser.read(column_data_types_config)
    return config_parser.get('ORIGINAL', 'categorical').split(',')


def get_additional_numerical():
    """Get all the numerical column names in additional data

    Returns
    -------
        List of strings
    """
    config_parser.read(column_data_types_config)
    return config_parser.get('ADDITIONAL', 'numerical').split(',')


def get_additional_categorical():
    """Get all the categorical column names in additional data

    Returns
    -------
        List of strings
    """
    config_parser.read(column_data_types_config)
    return config_parser.get('ADDITIONAL', 'categorical').split(',')

