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


data_names_config = '{}/main/config/data_file_names.cfg'.format(os.path.pardir)
column_data_types_config = '{}/main/config/column_data_types.cfg'.format(os.path.pardir)
activity_types_config = '{}/main/config/activity_types.cfg'.format(os.path.pardir)
pattern = re.compile("^\s+|\s*,\s*|\s+$")
parser_data_names, parser_column_types, parser_activity_types = ConfigParser(), ConfigParser(), ConfigParser()
parser_data_names.read(data_names_config)
parser_column_types.read(column_data_types_config)
parser_activity_types.read(activity_types_config)


def create_all_folders():
    """Create all the folders that are needed for the system"""
    pass


def get_fit_file_internal_args() -> [[]]:
    internal_args_all_athletes = []
    file_folders = os.listdir('{}/data'.format(os.path.pardir))
    for folder_name in file_folders:
        if folder_name.startswith('fit_'):
            if 'csv_{}'.format(folder_name[4:]) not in file_folders:
                internal_args = ['--subject-dir=../data',
                                 '--subject-name={}'.format('csv_{}'.format(folder_name[4:])),
                                 '--fit-source=../data/{}'.format(folder_name)]
                internal_args_all_athletes.append(internal_args)
    return internal_args_all_athletes


def get_all_spreadsheet_data_file_names():
    """Get all the spreadsheet data file names

    Returns
    -------
        List of strings
    """
    return [parser_data_names.get('SPREADSHEET-DATA-SETS', key) for key in list(parser_data_names['SPREADSHEET-DATA-SETS'].keys())]


def get_all_additional_data_folder_names():
    """Get all the additional data folder names

    Returns
    -------
        List of strings
    """
    return [parser_data_names.get('ADDITIONAL-DATA-FOLDERS', key) for key in list(parser_data_names['ADDITIONAL-DATA-FOLDERS'].keys())]


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
    if data_type == 'spreadsheet':
        if column_type == 'all':
            return ['Max Avg Power (20 min)', 'Avg Power', 'Avg Stroke Rate', 'Avg HR', 'Max HR', "Distance",
                    'Training Stress Score®', 'Total Strokes', 'Elev Gain', 'Elev Loss', 'Calories',
                    'Max Power', 'Max Speed', 'Avg Speed', 'Avg. Swolf', 'Avg Bike Cadence', 'Max Bike Cadence',
                    'Normalized Power® (NP®)', 'Number of Laps']
            # return pattern.split(config_parser.get('SPREADSHEET', 'numerical'))
    elif data_type == 'additional':
        if column_type == 'all':
            return pattern.split(parser_column_types.get('ADDITIONAL', 'numerical_ordered')) + \
                   pattern.split(parser_column_types.get('ADDITIONAL', 'numerical_fluctuating'))
        elif column_type == 'ordered':
            return pattern.split(parser_column_types.get('ADDITIONAL', 'numerical_ordered'))
        elif column_type == 'fluctuating':
            return pattern.split(parser_column_types.get('ADDITIONAL', 'numerical_fluctuating'))


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
    if data_type == 'spreadsheet':
        if column_type == 'all':
            return pattern.split(parser_column_types.get('SPREADSHEET', 'categorical'))
    elif data_type == 'additional':
        if column_type == 'all':
            return pattern.split(parser_column_types.get('ADDITIONAL', 'categorical'))
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
    if data_type == 'spreadsheet':
        return pattern.split(parser_activity_types.get('SPREADSHEET', 'activity_types'))
    elif data_type == 'additional':
        return pattern.split(parser_activity_types.get('ADDITIONAL', 'activity_types'))


def get_activity_subcategories(activity_type):
    if activity_type == 'running':
        return pattern.split(parser_activity_types.get('CATEGORIES', 'running'))
    elif activity_type == 'cycling':
        return pattern.split(parser_activity_types.get('CATEGORIES', 'cycling'))
    elif activity_type == 'swimming':
        return pattern.split(parser_activity_types.get('CATEGORIES', 'swimming'))
    elif activity_type == 'strength_training':
        return pattern.split(parser_activity_types.get('CATEGORIES', 'strength_training'))


def get_column_groups_for_imputation(data_type):
    if data_type == 'spreadsheet':
        return {'knn': {},
                'mice': {}}
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


def get_athlete_css(athletes_name) -> float:
    """ Get the critical swim speed of an athlete """
    with open(get_athlete_info_path(), 'r') as file:
        athletes_info_json = json.load(file)
    return athletes_info_json[athletes_name.title()]["critical swim speed"]


def get_athletes_lact_thr(athletes_name) -> (float, float):
    with open(get_athlete_info_path(), 'r') as file:
        athletes_info_json = json.load(file)
    jf_lact_thr = athletes_info_json[athletes_name.title()]["joe freil lactate threshold"]
    ac_lact_thr = athletes_info_json[athletes_name.title()]["andy coogan lactate threshold"]
    return (jf_lact_thr, ac_lact_thr)


class SystemReminder():

    def display_initialization_start(self):
        print("Initializing the System...")

    def display_initialization_end(self):
        print("System Initialized.")

    def display_data_cleaning_start(self, athletes_name, data_type):
        print("Cleaning {} {} data...".format(athletes_name, data_type))

    def display_data_cleaning_end(self, athletes_name, data_type):
        print("{} {} data done cleaning".format(athletes_name, data_type))

    def display_feature_engineering_start(self, athletes_name):
        print("Processing feature engineering on {} data...".format(athletes_name))

    def display_feature_engineering_end(self, athletes_name):
        print("Feature engineering on {} data is done.".format(athletes_name))

    def display_modeling_start(self, athletes_name, model_type):
        print("Building {} model on {} data...".format(model_type, athletes_name))

    def display_modeling_end(self, athletes_name, model_type):
        print("{} modeling on {} data is done.".format(model_type, athletes_name))


class FeatureManager():

    def get_all_features_for_modeling(self):
        return pattern.split(parser_activity_types.get('FEATURES', 'all'))

    def get_common_features_among_activities(self):
        return pattern.split(parser_activity_types.get('FEATURES', 'common'))

    def get_activity_specific_features(self, activity_type):
        try:
            return pattern.split(parser_activity_types.get('FEATURES', activity_type))
        except:
            print('No specific features for the given activity, return common features instead.')
            return self.get_common_features_among_activities()


if __name__ == '__main__':
    # The lines below are for test
    print(get_all_spreadsheet_data_file_names())
    print(get_all_additional_data_folder_names())
    print(get_athlete_css('eduardo oliveira'))
    print(get_activity_subcategories('swimming'))
    print(FeatureManager().get_all_features_for_modeling())
    print(get_fit_file_internal_args())


