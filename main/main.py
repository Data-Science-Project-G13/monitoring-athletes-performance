"""Main Script of the Project

This script allows the user to run the system.

This file can also be imported as a module and contains the following
functions:

    * main - the main function of the system
"""
# Self-defined modules
import system_initialize
import data_cleaning
import data_merge
import data_modelling


def main(athletes_name):
    """ Run the system
    Process: Data cleaning, Feature Engineering, Modeling.
    """
    athletes_name = athletes_name.lower()

    # Data Cleaning
    data_cleaning.main('spreadsheet', athletes_name)
    fit_activity_types, split_type = ['cycling', 'running', 'swimming'], 'real-time'
    for activity_type in fit_activity_types:
        data_cleaning.main('additional', athletes_name, activity_type=activity_type, split_type=split_type)

    # Feature Engineering
    data_merge.merge_spreadsheet_additional(athletes_name)

    # # Modeling
    # data_modelling.process_train_load_modeling(athletes_name)


if __name__ == '__main__':
    athletes_names = ['Eduardo Oliveira', 'Xu Chen', 'Carly Hart']
    system_initialize.initialize_system(athletes_names)
    for athletes_name in athletes_names:
        main(athletes_name)


