"""Main Script of the Project

This script allows the user to run the system.

This file can also be imported as a module and contains the following
functions:

    * main - the main function of the system
"""
# Self-defined modules
import os
import utility
import system_initialize
import data_cleaning
import data_merge
import data_modeling
from utility import SystemReminder as Reminder


def main(athletes_name: str,
         cleaning: bool=True,
         feature_engineering: bool=True,
         modeling: bool=True,
         pmc_generating: bool=True):
    """ Run the system
    Process: Data cleaning, Feature Engineering, Modeling.
    """
    Reminder().display_athlete_process_start(athletes_name)
    athletes_name = athletes_name.lower()

    # Data Cleaning
    if cleaning:
        spreadsheet_data_type, additional_data_type = 'spreadsheet', 'additional'
        show_spreadsheet_cleaning_details, show_additional_cleaning_details = True, True  # Change to True if would like to check details
        if utility.need_cleaning(athletes_name, spreadsheet_data_type, cleaning):
            utility.SystemReminder().display_data_cleaning_start(athletes_name, spreadsheet_data_type)
            data_cleaning.main('spreadsheet', athletes_name, verbose=show_spreadsheet_cleaning_details)
        utility.SystemReminder().display_data_cleaning_end(athletes_name, spreadsheet_data_type)
        if utility.need_cleaning(athletes_name, additional_data_type, cleaning):
            utility.SystemReminder().display_data_cleaning_start(athletes_name, additional_data_type)
            fit_activity_types, split_type = ['cycling', 'running', 'swimming', 'training'], 'real-time'
            for activity_type in fit_activity_types:
                data_cleaning.main('additional', athletes_name,
                                   activity_type=activity_type, split_type=split_type, verbose=show_additional_cleaning_details)
        utility.SystemReminder().display_data_cleaning_end(athletes_name, additional_data_type)


    # Feature Engineering
    system_initialize.initialize_features(athletes_name)
    if feature_engineering:
        show_feature_engineering_details = False
        utility.SystemReminder().display_feature_engineering_start(athletes_name)
        data_merge.process(athletes_name, verbose=show_feature_engineering_details)
        utility.SystemReminder().display_feature_engineering_end(athletes_name)


    # Modeling
    if modeling:
        Reminder().display_modeling_start(athletes_name, 'Assessing Training Load')
        data_modeling.process_train_load_modeling(athletes_name)
        Reminder().display_modeling_end(athletes_name, 'Assessing Training Load')

        Reminder().display_modeling_start(athletes_name, 'Predicting Performance')
        data_modeling.process_performance_modeling(athletes_name)
        Reminder().display_modeling_end(athletes_name, 'Predicting Performance')


    # Generate PMC
    if pmc_generating:
        pass

    Reminder().display_athlete_process_end(athletes_name)



if __name__ == '__main__':
    athletes_names = ['Eduardo Oliveira', 'Xu Chen', 'Carly Hart']
    do_initialize, do_cleaning, do_feature_engineering, do_modeling, do_pmc_generating = True, True, True, True, True
    if do_initialize:
        system_initialize.initialize_system()
    for athletes_name in athletes_names:
        main(athletes_name, do_cleaning, do_feature_engineering, do_modeling, do_pmc_generating)



