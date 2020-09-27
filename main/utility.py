"""System Utility

This file contains utility functions and classes for the system

This file can be imported as a module and contains the following functions:

    * create_all_folders - creates all folders that are necessary for the structure
    * get_fit_file_internal_args - get the arguments for converting fit files internally

    * need_clean - returns a boolean that indicates whether the athlete's data needs cleaning

    * get_all_spreadsheet_data_file_names - returns a list of spreadsheet data file names
    * get_all_additional_data_folder_names - returns a list of additional data folder names

    * get_spreadsheet_numerical - returns a list of numerical variables in spreadsheet data
    * get_spreadsheet_categorical - returns a list of categorical variables in spreadsheet data
    * get_additional_numerical - returns a list of numerical variables in additional data
    * get_additional_categorical - returns a list of categorical variables in additional data

    * get_spreadsheet_activity_types - returns a list of activity types in spreadsheet data
    * get_additional_activity_types - returns a list of activity types in additional data

This file can be imported as a module and contains the following classes:
    * SystemReminder - displays reminders while running the system
    * FeatureManager - manages the features

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
    folders = {'{}/data/'.format(os.path.pardir),
               '{}/data/cleaned_additional/'.format(os.path.pardir),
               '{}/data/cleaned_spreadsheet/'.format(os.path.pardir),
               '{}/log/'.format(os.path.pardir),
               '{}/models/'.format(os.path.pardir),
               '{}/plots/'.format(os.path.pardir)}
    for folder in folders:
        print('creating {}'.format(folder))
        if not os.path.exists(folder):
            os.mkdir(folder)


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


def check_data_existence(athletes_name):
    spreadsheet_exists, additional_exists = False, False
    if os.path.exists('{}/data/{}.csv'.format(os.path.pardir, athletes_name.title())):
        spreadsheet_exists = True
    if os.path.exists('{}/data/fit_{}'.format(os.path.pardir, '_'.join(athletes_name.split()))):
        additional_exists = True
    return spreadsheet_exists, additional_exists


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


def need_cleaning(athletes_name: str, data_type: str, do_cleaning: bool) -> bool:
    if data_type == 'spreadsheet':
        if '{}.csv'.format(athletes_name.title()) in os.listdir('{}/data/cleaned_spreadsheet'.format(os.path.pardir))\
                and not do_cleaning:
            return False
    elif data_type == 'additional':
        if '_'.join(athletes_name.lower().split()) in os.listdir('{}/data/cleaned_additional'.format(os.path.pardir)) \
                and not do_cleaning:
            return False
    return True


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

    line = '='*25

    def display_initialization_start(self):
        print("Initializing the System...")

    def display_initialization_end(self):
        print("System Initialized.")

    def display_athlete_process_start(self, athletes_name):
        print("\n{} The Start of {} Athlete Analytics {}".format(self.line, athletes_name.title(), self.line))

    def display_athlete_process_end(self, athletes_name):
        print("{} The End of {} Athlete Analytics {}\n".format(self.line, athletes_name.title(), self.line))

    def raise_athete_spreadsheet_doesnt_exist(self, athletes_name):
        raise Exception("Couldn't Find Athlete {}'s Spreadsheet Data.".format(athletes_name.title()))

    def display_athete_additional_doesnt_exist(self, athletes_name):
        print("Couldn't Find Athlete {}'s Fit File Data Folder.".format(athletes_name.title()))

    def display_fit_file_converted(self, athletes_name):
        print("{} fit files have converted to csv files.".format(athletes_name.title()))

    def display_data_cleaning_start(self, athletes_name, data_type):
        print("Cleaning {} {} data...".format(athletes_name.title(), data_type))

    def display_data_cleaning_end(self, athletes_name, data_type):
        print("{} {} data done cleaning".format(athletes_name.title(), data_type))

    def display_feature_engineering_start(self, athletes_name):
        print("Processing feature engineering on {} data...".format(athletes_name.title()))

    def display_feature_engineering_end(self, athletes_name):
        print("Feature engineering on {} data is done.".format(athletes_name.title()))

    def display_modeling_start(self, athletes_name, model_type):
        print("Building {} model on {} data...".format(model_type, athletes_name.title()))

    def display_modeling_end(self, athletes_name, model_type):
        print("{} modeling on {} data is done.".format(model_type, athletes_name.title()))


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


