import os
import sys
import re
import datetime
import logging
import matplotlib.pyplot as plt
from statistics import mean, median
from data_loader import DataLoader


def _setup_logger():
    global logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename='logginng')
    logger = logging.getLogger(__name__)


class SummaryGenerater():

    def __init__(self, file_name, athlete_dataframe):
        self.athlete_dataframe = athlete_dataframe
        self.file_name = file_name

    def summarize_sport_categories(self):
        print('====== Titles and Activity Types ======')
        for title in self.athlete_dataframe['Title'].value_counts().index:
            print(tuple(self.athlete_dataframe[self.athlete_dataframe['Title'] == title][['Title', 'Activity Type']].iloc[0]))
        print('\n')
        print('====== Activity Type and Counts ======')
        print(self.athlete_dataframe['Activity Type'].value_counts())

    def summarize_dates(self, activity_type):
        '''
        Dates format are so messy !!!
        :param activity_type:
        :return:
        '''
        try:
            df = self.athlete_dataframe[self.athlete_dataframe['Activity Type'] == activity_type]
            dates = [datetime.datetime.strptime(date, "%d-%m-%y %H:%M") for date in df['Date']]
            dates.sort()
            sorted_dates = [datetime.datetime.strftime(date, "%d-%m-%y %H:%M") for date in dates]
            print('Record from {} to {}'.format(sorted_dates[0], sorted_dates[-1:][0]))
        except Exception as e:
            logger.exception('FILE {}: '.format(self.file_name), e)

    def summarize_workout_time(self, activity_type):
        try:
            df = self.athlete_dataframe[self.athlete_dataframe['Activity Type'] == activity_type]
            total_seconds_list = []
            for time in df['Time']:
                h_m_s = time.split(':')
                total_seconds = 0
                for t in h_m_s:
                    total_seconds = total_seconds*60 + float(t)
                    total_seconds_list.append(total_seconds)
            print('Max workout time: {} seconds'.format(max(total_seconds_list)))
            print('Min workout time: {} seconds'.format(min(total_seconds_list)))
            print('Mean workout time: {} seconds'.format(round(mean(total_seconds_list)), 2))
            print('Median workout time: {} seconds'.format(median(total_seconds_list)))
            # plt.boxplot(total_seconds_list)
            # plt.show()
        except Exception as e:
            logger.exception('FILE {}: '.format(self.file_name), e)

    def summarize_features(self, activity_type):
        features = ['Distance', 'Avg HR', 'Avg Speed', 'Calories', 'Training Stress Score®', 'Avg Power']
        df = self.athlete_dataframe[self.athlete_dataframe['Activity Type'] == activity_type]
        for feature in features:
            try:
                feature_values = [float(str(var).replace(',', '')) for var in df[feature] if re.match(r'^-?\d+(?:\.\d+)?$', str(var)) is not None]
                if len(feature_values) != 0:
                    print('--- {} Summary ---'.format(feature))
                    print('Max {}: {}'.format(feature, max(feature_values)))
                    print('Min {}: {}'.format(feature, min(feature_values)))
                    print('Mean {}: {}'.format(feature, round(mean(feature_values)), 2))
                    print('Median {}: {}'.format(feature, median(feature_values)))
            except Exception as e:
                logger.exception('FILE: {}'.format(self.file_name), e)


def _process_summarizing(file_name):
    athlete_dataframe = DataLoader(file_name).load_athlete_dataframe()
    generator = SummaryGenerater(file_name, athlete_dataframe)
    generator.summarize_sport_categories()
    print('\n')
    activity_types = generator.athlete_dataframe['Activity Type'].value_counts().index[0:4]
    for activity_type in activity_types:
        print('====== {} ====== '.format(activity_type))
        print('--- Date Summary ---')
        generator.summarize_dates(activity_type)
        print('--- Workout Time Summary ---')
        generator.summarize_workout_time(activity_type)
        generator.summarize_features(activity_type)
        print('\n')


if __name__ == '__main__':
    _setup_logger()
    data_path = '{}/data'.format(os.path.pardir)
    dirs = os.listdir(data_path)
    for file_name in dirs:
        if file_name.endswith(".csv"):
            sys.stdout = open('{}/data_summaries/{} Summary.txt'.format(os.path.pardir, file_name), 'w')
            _process_summarizing(file_name)
            sys.stdout.close()




