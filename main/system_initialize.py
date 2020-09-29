"""System Initialize

This script allows the user to initialize the analysis system.

This file can also be imported as a module and contains the following
functions:

    * create_directory_structures
    * convert_fit_files_to_csv
    * initialize_critical_swim_speed
    * initialize_lactate_threshold
    * initialize_system
    * initialize_features
"""


# Packages
import os
import json
import pandas as pd
from data_loader import DataLoader
# Self-defined modules
import utility
from fit_file_convert import process_all
athlete_info_json_path = utility.get_athlete_info_path()


def create_directory_structures():
    """ Create all the directories and folders that are needed for the project
    """
    utility.create_all_folders()


def convert_fit_files_to_csv():
    for internal_args in utility.get_fit_file_internal_args():
        process_all.main(internal_args=internal_args)
        utility.SystemReminder().display_fit_file_converted(internal_args[1].split('=')[1][4:])


def initialize_configurations():
    """ Create all files, config and json, that are needed for the project
    """
    initialize_json()
    initialize_config()


def initialize_config():
    pass


def initialize_json():
    athletes_names = [file_name[:-4] for file_name in os.listdir('{}/data'.format(os.pardir)) if file_name.endswith('.csv') ]
    if not os.path.exists(athlete_info_json_path):
        with open(athlete_info_json_path, 'w') as file:
            json.dump({}, file, indent=4)
    with open(athlete_info_json_path, 'r') as file:
        athletes_info_json = json.load(file)
        for athletes_name in athletes_names:
            if athletes_name.title() not in athletes_info_json:
                athletes_info_json[athletes_name.title()] = {
                    "athlete type": None,
                    "gender": None,
                    "age": None,
                    "height": None,
                    "pre weight": None,
                    "post weight": None,
                    "injuries": None,
                    "critical swim speed": None,
                    "joe freil lactate threshold": None,
                    "andy coogan lactate threshold": None,
                    "training load best models": {"running": None, "swimming": None, "cycling": None,
                                                 "strength training": None, "others": None},
                    "performance best models": {"running": None, "swimming": None, "cycling": None}
                }
    with open(athlete_info_json_path, 'w') as file:
        json.dump(athletes_info_json, file, indent=4)


def initialize_critical_swim_speed(athletes_name: str):
    ''' Initialize the critical swim speed of an athlete
    Parameters
    ----------
    athletes_name: str
    '''
    data_loader_additional = DataLoader('additional')
    cleaned_additional_data_filenames = data_loader_additional.load_cleaned_additional_data(athletes_name)

    def _calculate_css():
        first_50m_distance, first_50m_times, first_400m_distance, first_400m_times = [], [], [], []
        for file_name in [file_name for file_name in cleaned_additional_data_filenames if 'swimming' in file_name]:
            df = pd.read_csv(file_name)
            time, distance = list(df['time_in_seconds']), list(df['distance'])
            for i, d in enumerate(distance):
                distance_in_meters = int((d - distance[0]) * 1000)
                if abs(distance_in_meters-50) < 5:
                    first_50m_times.append(time[i]-time[0])
                    first_50m_distance.append(distance_in_meters)
                if abs(distance_in_meters-400) < 10:
                    first_400m_times.append(time[i]-time[0])
                    first_400m_distance.append(distance_in_meters)
        mean_dis_diff = sum(first_400m_distance)/len(first_400m_distance) - sum(first_50m_distance)/len(first_50m_distance)
        mean_time_diff = sum(first_400m_times)/len(first_400m_times) - sum(first_50m_times)/len(first_50m_times)
        return (mean_dis_diff/mean_time_diff)

    return _calculate_css()


def initialize_lactate_threshold(athletes_name: str):
    ''' Initialize the lactate threshold of an athlete
    Parameters
    ----------
    athletes_name: str
    '''
    data_loader_additional = DataLoader('additional')
    cleaned_additional_data_filenames = data_loader_additional.load_cleaned_additional_data(athletes_name)

    def _calculate_Joe_Freil_lactate_threshold():
        heart_rate_entry=[]
        for file_name in [file_name for file_name in cleaned_additional_data_filenames if 'running' in file_name]:
            temporary_list=[]
            df = pd.read_csv(file_name)
            if df['heart_rate'].isnull().values.any()==True:
                continue
            time, heart_rate = list(df['time_in_seconds']), list(df['heart_rate'])
            for i,t in enumerate(time):
                time_difference_current=time[i]-time[0]
                if 600 <= time_difference_current <= 1805:
                    temporary_list.append(heart_rate[i])
                    if abs(time_difference_current-1805)<5:
                        heart_rate_entry=heart_rate_entry+temporary_list
                        break
        average_heart_rate=sum(heart_rate_entry)/len(heart_rate_entry)
        return average_heart_rate

    def _calculate_Andy_Coogan__lactate_threshold():
        heart_rate_entry = []
        for file_name in [file_name for file_name in cleaned_additional_data_filenames if 'running' in file_name]:
            temporary_list = []
            df = pd.read_csv(file_name)
            if df['heart_rate'].isnull().values.any() == True:
                continue
            time, heart_rate = list(df['time_in_seconds']), list(df['heart_rate'])
            for i, t in enumerate(time):
                time_difference_current = time[i] - time[0]
                if time_difference_current <= 3605:
                    temporary_list.append(heart_rate[i])
                    if abs(time_difference_current - 3605) < 5:
                        heart_rate_entry = heart_rate_entry + temporary_list
                        break
        average_heart_rate = sum(heart_rate_entry) / len(heart_rate_entry)
        return average_heart_rate

    return _calculate_Joe_Freil_lactate_threshold(), _calculate_Andy_Coogan__lactate_threshold()


def initialize_system():
    """ Initialize the whole system
    """
    utility.SystemReminder().display_initialization_start()
    create_directory_structures()
    convert_fit_files_to_csv()
    initialize_configurations()
    utility.SystemReminder().display_initialization_end()


def initialize_characteristics(athletes_name):
    with open(athlete_info_json_path, 'r') as file:
        athletes_info_json = json.load(file)
    athletes_css = initialize_critical_swim_speed(athletes_name)
    athletes_info_json[athletes_name.title()]["critical swim speed"] = athletes_css
    jf_threshold, ac_threshold = initialize_lactate_threshold(athletes_name)
    athletes_info_json[athletes_name.title()]["joe freil lactate threshold"] = jf_threshold
    athletes_info_json[athletes_name.title()]["andy coogan lactate threshold"] = ac_threshold
    with open(athlete_info_json_path, 'w') as file:
        json.dump(athletes_info_json, file, indent=4)



if __name__ == '__main__':
    athletes_names = ['Eduardo Oliveira', 'Xu Chen', 'Carly Hart']
    # initialize_system()
    # initialize_characteristics(athletes_names[0])
    initialize_configurations()

