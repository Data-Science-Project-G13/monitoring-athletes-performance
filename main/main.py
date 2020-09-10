"""Main Script of the Project

This script allows the user to run the system.

This file can also be imported as a module and contains the following
functions:

    * main - the main function of the system
"""

import system_initialize
import data_cleaning
import data_feature_engineering
import data_merge
import data_modeling


def main(athletes_name):
    """ Run the system
    Process: Data cleaning, Feature Engineering, Modeling.
    """

    # Data Cleaning
    data_cleaning.main('spreadsheet', athletes_name)
    activity_types, split_type = ['cycling', 'running', 'swimming'], 'real-time'
    for activity_type in activity_types:
        data_cleaning.main('additional', athletes_name, activity_type=activity_type, split_type=split_type)

    # Feature Engineering
    data_merge.merge_spreadsheet_additional(athletes_name)

    # Modeling
    data_modeling.process_train_load_modeling(athletes_name)


if __name__ == '__main__':
    system_initialize.initialize_system()
    athletes_names = ['Eduardo Oliveira']   # 'Simon R Gronow'
    for athletes_name in athletes_names:
        main(athletes_name)


