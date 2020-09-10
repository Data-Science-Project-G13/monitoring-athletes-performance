import os
import json
import pandas as pd
from data_loader import DataLoader


def create_directory_structures():
    """
    Create all the directories and folders that are needed for the project
    :return:
    """
    pass


def initialize_critical_swimming_speed(athletes_name: str):
    '''
    Parameters
    ----------
    athletes_name
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
        mean_dis_diff = sum(first_400m_distance) / len(first_400m_distance) - sum(first_50m_distance)/len(first_50m_distance)
        mean_time_diff = sum(first_400m_times)/len(first_400m_times) - sum(first_50m_times)/len(first_50m_times)
        return mean_dis_diff/mean_time_diff

    athletes_css = _calculate_css()
    with open('config/athlete_personal_info.json', 'r') as file:
        athletes_info_json = json.load(file)
    athletes_info_json[athletes_name.title()]["critical swim speed"] = athletes_css
    with open('config/athlete_personal_info.json', 'w') as file:
        json.dump(athletes_info_json, file, indent = 4)


def initialize_lactate_threshold(athletes_name: str):
    pass



def initialize_system():
    create_directory_structures()
    athletes = ['eduardo oliveira']
    for athlete in athletes:
        initialize_critical_swimming_speed(athlete)
        initialize_lactate_threshold(athlete)

if __name__ == '__main__':
    initialize_system()
