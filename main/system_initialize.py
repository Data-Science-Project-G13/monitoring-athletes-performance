# Packages
import json
import pandas as pd
from data_loader import DataLoader
# Self-defined modules
import utility
athlete_info_json_path = utility.get_athlete_info_path()


def create_directory_structures():
    """ Create all the directories and folders that are needed for the project
    """
    pass


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
        return mean_dis_diff/mean_time_diff

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
            time, heart_rate = list(df['time_in_seconds']), list(df['heart_rate'])
            for i,t in enumerate(time):
                time_difference_current=time[i]-time[0]
                time_difference_next=time[i+1]-time[0]
                if 600 <= time_difference_current <= 1805:
                    temporary_list.append(heart_rate)
                    if time_difference_next>1805:
                        heart_rate_entry=heart_rate_entry+temporary_list
                        break
        average_heart_rate=sum(heart_rate_entry)/len(heart_rate_entry)
        return average_heart_rate


    return _calculate_Joe_Freil_lactate_threshold()


def initialize_system():
    """ Initialize the whole system
    """
    create_directory_structures()
    athletes = ['eduardo oliveira']
    for athletes_name in athletes:
        with open(athlete_info_json_path, 'r') as file:
            athletes_info_json = json.load(file)
        json_changes_made = False
        if not athletes_info_json[athletes_name.title()]["critical swim speed"]:
            athletes_css = initialize_critical_swim_speed(athletes_name)
            athletes_info_json[athletes_name.title()]["critical swim speed"] = athletes_css
            json_changes_made = True
        if not athletes_info_json[athletes_name.title()]["lactate threshold"]:
            lactate_threshold=initialize_lactate_threshold(athletes_name)
            athletes_info_json[athletes_name.title()]["lactate threshold"]=lactate_threshold
            json_changes_made = True
        if json_changes_made:
            with open(athlete_info_json_path, 'w') as file:
                json.dump(athletes_info_json, file, indent=4)


if __name__ == '__main__':
    initialize_system()
