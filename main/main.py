"""Main Script of the Project

This script allows the user to run the system.

This file can also be imported as a module and contains the following
functions:

    * main - the main function of the system
"""
# Self-defined modules
import utility
import system_initialize
import data_cleaning
import data_merge
import data_modeling
from utility import SystemReminder as Reminder


def main(athletes_name):
    """ Run the system
    Process: Data cleaning, Feature Engineering, Modeling.
    """
    athletes_name = athletes_name.lower()

    # Data Cleaning
    spreadsheet_data_type, additional_data_type = 'spreadsheet', 'additional'
    show_spreadsheet_cleaning_details, show_additional_cleaning_details = False, False  # Change to True if would like to check cleaning details
    utility.SystemReminder().display_data_cleaning_start(athletes_name, spreadsheet_data_type)
    data_cleaning.main('spreadsheet', athletes_name, verbose=show_spreadsheet_cleaning_details)
    utility.SystemReminder().display_data_cleaning_end(athletes_name, spreadsheet_data_type)

    utility.SystemReminder().display_data_cleaning_start(athletes_name, additional_data_type)
    fit_activity_types, split_type = ['cycling', 'running', 'swimming', 'training'], 'real-time'
    for activity_type in fit_activity_types:
        data_cleaning.main('additional', athletes_name,
                           activity_type=activity_type, split_type=split_type, verbose=show_additional_cleaning_details)
    utility.SystemReminder().display_data_cleaning_end(athletes_name, additional_data_type)


    # Feature Engineering
    show_feature_engineering_details = False
    utility.SystemReminder().display_feature_engineering_start(athletes_name)
    data_merge.process(athletes_name, verbose=show_feature_engineering_details)
    utility.SystemReminder().display_feature_engineering_end(athletes_name)


    # Modeling
    Reminder().display_modeling_start(athletes_name, 'Assessing Training Load')
    data_modeling.process_train_load_modeling(athletes_name)
    Reminder().display_modeling_end(athletes_name, 'Assessing Training Load')

    Reminder().display_modeling_start(athletes_name, 'Predicting Performance')
    data_modeling.process_performance_modeling(athletes_name)
    Reminder().display_modeling_end(athletes_name, 'Predicting Performance')


if __name__ == '__main__':
    athletes_names = ['Eduardo Oliveira', 'Xu Chen', 'Carly Hart']
    system_initialize.initialize_system(athletes_names)
    for athletes_name in athletes_names:
        main(athletes_name)



