""" Main Script of the Project

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
import generate_pmc
import system_argparse
from utility import SystemReminder as Reminder


def main(internal_args=None):
    options = system_argparse.parse_options(internal_args)
    if options['do_initialize']:
        system_initialize.initialize_system()
    for athletes_name in options['athletes_names']:
        run_system(athletes_name=athletes_name.lower().replace('_', ' '),
                   cleaning=options['do_cleaning'],
                   feature_engineering=options['do_feature_engineering'],
                   modeling=options['do_modeling'],
                   pmc_generating=options['do_pmc_generating'])


def run_system(athletes_name: str, cleaning: bool=True, feature_engineering: bool=True,
               modeling: bool=True, pmc_generating: bool=True):
    """ Run the system
    Process: Data cleaning, Feature Engineering, Modeling.
    """
    reminder = Reminder()
    reminder.display_athlete_process_start(athletes_name)
    athletes_name = athletes_name.lower()
    spreadsheet_exists, additional_exists = utility.check_data_existence(athletes_name)
    if not spreadsheet_exists:
        reminder.raise_athete_spreadsheet_doesnt_exist(athletes_name)
    if not additional_exists:
        reminder.display_athete_additional_doesnt_exist(athletes_name)

    # Data Cleaning
    if cleaning:
        spreadsheet_data_type, additional_data_type = 'spreadsheet', 'additional'
        show_spreadsheet_cleaning_details, show_additional_cleaning_details = True, True  # Change to True if would like to check details
        if utility.need_cleaning(athletes_name, spreadsheet_data_type, cleaning):
            reminder.display_data_cleaning_start(athletes_name, spreadsheet_data_type)
            data_cleaning.main('spreadsheet', athletes_name, verbose=show_spreadsheet_cleaning_details)
        reminder.display_data_cleaning_end(athletes_name, spreadsheet_data_type)
        if utility.need_cleaning(athletes_name, additional_data_type, cleaning):
            reminder.display_data_cleaning_start(athletes_name, additional_data_type)
            fit_activity_types, split_type = ['cycling', 'running', 'swimming', 'training'], 'real-time'
            for activity_type in fit_activity_types:
                data_cleaning.main('additional', athletes_name, activity_type=activity_type,
                                   split_type=split_type, verbose=show_additional_cleaning_details)
        reminder.display_data_cleaning_end(athletes_name, additional_data_type)

    # Feature Engineering
    if additional_exists:
        system_initialize.initialize_characteristics(athletes_name)
    if feature_engineering:
        show_feature_engineering_details = False
        reminder.display_feature_engineering_start(athletes_name)
        data_merge.process(athletes_name, verbose=show_feature_engineering_details)
        reminder.display_feature_engineering_end(athletes_name)

    # Modeling
    if modeling:
        reminder.display_modeling_start(athletes_name, 'Assessing Training Load')
        data_modeling.process_train_load_modeling(athletes_name)
        reminder.display_modeling_end(athletes_name, 'Assessing Training Load')

        # reminder.display_modeling_start(athletes_name, 'Predicting Performance')
        # data_modeling.process_performance_modeling(athletes_name)
        # reminder.display_modeling_end(athletes_name, 'Predicting Performance')

    # Generate PMC
    if pmc_generating:
        reminder.display_pmc_generation_start(athletes_name)
        generate_pmc.process_pmc_generation(athletes_name, save_pmc_figure=False)
        reminder.display_pmc_generation_end(athletes_name)

    reminder.display_athlete_process_end(athletes_name)


if __name__ == '__main__':
    # For Users
    # main()

    # For Developers. Uncomment the following lines if you would like test the system.
    athletes_names = ['eduardo_oliveira', 'xu_chen', 'carly_hart']
    internal_args = ['--athletes-names={}'.format(' '.join(athletes_names)),
                     '--initialize-system=False', '--clean-data=False', '--process-feature-engineering=False',
                     '--build-model=False', '--generate-pmc=True']
    main(internal_args)

