"""Athletes Data Loader

This script contains common functions that support the project.

This script requires that `pandas` be installed.

This file can be imported as a module and contains the following
functions:

    * get_all_spreadsheet_data_file_names - return a list of spreadsheet data file names
    * get_all_additional_data_folder_names - return a list of additional data folder names

    * get_spreadsheet_numerical - returns a list of numerical variables in spreadsheet data
    * get_spreadsheet_categorical - returns a list of categorical variables in spreadsheet data
    * get_additional_numerical - returns a list of numerical variables in additional data
    * get_additional_categorical - returns a list of categorical variables in additional data

    * get_spreadsheet_activity_types - returns a list of activity types in spreadsheet data
    * get_additional_activity_types - returns a list of activity types in additional data

"""

import os
import re
import json
from configparser import ConfigParser


config_parser = ConfigParser()
data_names_config = '{}/main/config/data_file_names.cfg'.format(os.path.pardir)
column_data_types_config = '{}/main/config/column_data_types.cfg'.format(os.path.pardir)
activity_types_config = '{}/main/config/activity_types.cfg'.format(os.path.pardir)
pattern = re.compile("^\s+|\s*,\s*|\s+$")


def create_all_folders():
    """Create all the folders that are needed for the system"""
    pass


def get_all_spreadsheet_data_file_names():
    """Get all the spreadsheet data file names

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


def get_numerical_columns(data_type, column_type='all'):
    """Get all the numerical column names in data

    Parameters
    -------
    data_type: str
        The type of the data set.
    column_type: str
        The type of the column

    Returns
    -------
        List of strings
    """
    config_parser.read(column_data_types_config)
    if data_type == 'spreadsheet':
        if column_type == 'all':
            return ['Max Avg Power (20 min)', 'Avg Power', 'Avg Stroke Rate', 'Avg HR', 'Max HR', "Distance",
                    'Training Stress Score®', 'Total Strokes', 'Elev Gain', 'Elev Loss', 'Calories',
                    'Max Power', 'Max Speed', 'Avg Speed', 'Avg. Swolf', 'Avg Bike Cadence', 'Max Bike Cadence',
                    'Normalized Power® (NP®)', 'Number of Laps']
            # return pattern.split(config_parser.get('SPREADSHEET', 'numerical'))
    elif data_type == 'additional':
        if column_type == 'all':
            return pattern.split(config_parser.get('ADDITIONAL', 'numerical_ordered')) + \
                   pattern.split(config_parser.get('ADDITIONAL', 'numerical_fluctuating'))
        elif column_type == 'ordered':
            return pattern.split(config_parser.get('ADDITIONAL', 'numerical_ordered'))
        elif column_type == 'fluctuating':
            return pattern.split(config_parser.get('ADDITIONAL', 'numerical_fluctuating'))


def get_categorical_columns(data_type, column_type='all'):
    """Get all the numerical column names in data

    Parameters
    -------
    data_type: str
        The type of the data set.

    Returns
    -------
        List of strings
    """
    config_parser.read(column_data_types_config)
    if data_type == 'spreadsheet':
        if column_type == 'all':
            return pattern.split(config_parser.get('SPREADSHEET', 'categorical'))
    elif data_type == 'additional':
        if column_type == 'all':
            return pattern.split(config_parser.get('ADDITIONAL', 'categorical'))
        elif column_type == 'ordinal':
            return
        elif column_type == 'non-ordinal':
            return


def get_all_columns(data_type):
    """Get all the column names in spreadsheet data

    Parameters
    -------
    data_type: str
        The type of the data set.

    Returns
    -------
    columns: List of strings
        The list of all column names
    """
    columns = get_numerical_columns(data_type) + get_categorical_columns(data_type)
    return columns


def get_activity_types(data_type):
    """Get all the activity types in the data

    Returns
    -------
        List of strings
    """
    config_parser.read(activity_types_config)
    if data_type == 'spreadsheet':
        return pattern.split(config_parser.get('SPREADSHEET', 'activity_types'))
    elif data_type == 'additional':
        return pattern.split(config_parser.get('ADDITIONAL', 'activity_types'))


def get_column_groups_for_imputation(data_type):
    if data_type == 'spreadsheet':
        return {}
    elif data_type == 'additional':
        return {'univariate': {'timezone'},
                'multivariate1': {"distance", "timestamp"},
                'multivariate2': {"speed", "heart_rate", "cadence"},
                'interpolation': {"position_lat", "position_long", "altitude"},
                'regression': {"speed", "heart_rate", "cadence"}}


def get_outlier_color_labels_additional():
    return ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'grey', 'olive', 'cyan', 'black',
             'maroon', 'chocolate', 'gold', 'yellow', 'lawngreen', 'aqua', 'steelblue', 'navy',
             'indigo', 'magenta', 'crimson', 'red']


def get_athlete_info_path():
    return '{}/main/config/athlete_personal_info.json'.format(os.path.pardir)


def get_athlete_css(athletes_name):
    """ Get the critical swim speed of an athlete """
    with open(get_athlete_info_path(), 'r') as file:
        athletes_info_json = json.load(file)
    return athletes_info_json[athletes_name.title()]["critical swim speed"]



if __name__ == '__main__':
    # The lines below are for test
    print(get_all_spreadsheet_data_file_names())
    print(get_all_additional_data_folder_names())
    print(get_athlete_css('eduardo oliveira'))


