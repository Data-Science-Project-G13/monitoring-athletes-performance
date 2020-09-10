import pandas as pd
from data_loader import DataLoader


def create_directory_structures():
    """
    Create all the directories and folders that are needed for the project
    :return:
    """
    pass


def get_critical_swimming_speed(athletes_name: str):
    '''

    Parameters
    ----------
    athlete_df

    Returns
    -------

    '''
    data_loader_additional = DataLoader('additional')
    cleaned_additional_data_filenames = data_loader_additional.load_cleaned_additional_data(athletes_name)
    first_50m_speeds, first_400m_speeds = [], []
    for file_name in [file_name for file_name in cleaned_additional_data_filenames if 'swimming' in file_name]:
        df = pd.read_csv(file_name)


def initialize_system():
    create_directory_structures()
    athletes = ['eduardo oliveira']
    for athlete in athletes:
        get_critical_swimming_speed(athlete)
