"""Make Plots

This script allows the user to make plots from the athletes data.

This tool accepts comma separated value files (.csv).

This script requires that `pandas`, `matplotlib.pyplot` be installed
within the Python environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * plot_PMC - plots a PMC for a certain athlete
    * plot_valid_TSS_pie
    * plot_activity_tendency_bar
    * plot_athlete_level_pie
    * plot_boxplots
    * plot_frequency

"""

import os
import sys
import re
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statistics
import numpy as np
from data_loader import DataLoader
from data_preprocess import DataPreprocessor


# Set the data frame display option
pd.set_option('display.max_row', 20)
pd.set_option('display.max_columns', 4)


def create_plot_folder():
    """Create the plot folder if it does not exist
    """
    plot_folder = '{}/plots'.format(os.path.pardir)
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)


class SingleAthleteDataPlotter():
    """
    A class used to make plots related to a certain athlete

    ...

    Attributes
    ----------
    file_name : str
        The file name of the athlete's data

    Methods
    -------
    plot_TSS()
        Plot TSS points
    plot_fatigue_and_fitness()
        Plot Fatigue, Fitness and Form
    plot_PMC()
        Plot the PMC
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.athlete_dataframe = self.preprocess()
        self.fig, self.ax = plt.subplots()
        self.ax2 = self.ax.twinx()

    def preprocess(self):
        """Preprocess the athlete's data frame

        Returns
        -------
        preprocessed_df: pandas data frame
            A data frame that is able to generate PMC
        """
        athlete_dataframe = DataLoader(self.file_name).load_athlete_dataframe()
        preprocessor = DataPreprocessor(athlete_dataframe)
        activity_types = preprocessor.get_activity_types()
        for activity_type in activity_types:
            preprocessor.fill_out_tss(activity_type)
        preprocessed_df = preprocessor.athlete_dataframe
        preprocessed_df['Date'] = pd.to_datetime(preprocessed_df.Date, dayfirst=True)
        preprocessed_df = preprocessed_df.sort_values(by=['Date'], ascending=True)
        return preprocessed_df

    def plot_TSS(self):
        """Plot TSS
        """
        df = self.athlete_dataframe
        dates = df['Date'] #mdates.num2date(mdates.datestr2num(df['Date']))
        tss = df['Training Stress Score®']
        tss = pd.DataFrame([float(str(var).replace(',', '')) for var in tss])

        self.ax.plot(dates, tss, '.', label='TSS', color='lightskyblue')
        self.ax.legend(loc=2)
        # rotate and align the tick labels so they look better
        self.fig.autofmt_xdate()
        # self.ax.yaxis.set_label_position("left")
        # self.ax.set_xlabel('Dates')
        self.ax.set_ylabel('TSS')

    def plot_fatigue_and_fitness(self):
        """Plot Fatigue, Fitness and Form
        """
        df = self.athlete_dataframe[['Date', 'Training Stress Score®']]
        df = df[df['Training Stress Score®'] != 0]
        dates = df['Date']
        df = df.set_index('Date')
        df['ATL'] = df.rolling('7d', min_periods=1)['Training Stress Score®'].mean()
        df['ATL2'] = df.rolling(7, min_periods=1)['Training Stress Score®'].mean()
        df['CTL'] = df.rolling('42d', min_periods=1)['Training Stress Score®'].mean()
        # print(df[['Training Stress Score®', 'ATL', 'ATL2']])
        l1 = self.ax2.plot(dates, df['ATL'], label='Fatigue (ATL)', color='green')
        l2 = self.ax2.plot(dates, df['CTL'], label='Fitness (CTL)', color='orange')
        l3 = self.ax2.plot(dates, df['CTL'] - df['ATL'], label='Form (TSB)', color='pink')
        # lns = dot1 + l1 + l2 + l3
        # labs = [l.get_label() for l in lns]
        self.ax2.legend(loc=0)
        plt.axhline(xmin=0.05, xmax=1, y=-30, color='r', linestyle='-')
        plt.axhline(xmin=0.05, xmax=1, y=-10, color='g', linestyle='-')
        plt.axhline(xmin=0.05, xmax=1, y=5, color='grey', linestyle='-')
        plt.axhline(xmin=0.05, xmax=1, y=25, color='royalblue', linestyle='-')
        # TODO: Generalize the line ends
        plt.text(x=datetime.date(2019, 8, 1), y=-40, s='High Risk Zone', horizontalalignment='right', color='red')
        plt.text(x=datetime.date(2019, 8, 1), y=-20, s='Optimal Training Zone', horizontalalignment='right', color='green')
        plt.text(x=datetime.date(2019, 8, 1), y=-4, s='Grey Zone', horizontalalignment='right', color='grey')
        plt.text(x=datetime.date(2019, 8, 1), y=15, s='Freshness Zone', horizontalalignment='right', color='royalblue')
        plt.text(x=datetime.date(2019, 8, 1), y=32, s='Transition Zone', horizontalalignment='right', color='darkgoldenrod')
        self.ax2.set_ylabel('CTL / ATL / TSB')

    def plot_PMC(self):
        """Plot the PMC
        Show the plot or save the plot to the plots folder
        """
        self.plot_TSS()
        self.plot_fatigue_and_fitness()
        plt.title('Performance Management Chart - {}'.format(self.file_name.split('.')[0]))
        plt.legend()
        # plt.show()
        plt.savefig('{}/plots/PMC - {}.jpg'.format(os.path.pardir, self.file_name.split('.')[0]), format='jpg', dpi=1200)


def plot_PMC():
    data_path = '{}/data'.format(os.path.pardir)
    dirs = os.listdir(data_path)
    for file_name in dirs:
        # if file_name.endswith(".csv"):
        if file_name == 'Simon R Gronow (Novice).csv':
            pass


class MultipleAtheletesDataPlotter():

    def __init__(self):
        self.novice_dict = {'Data_amount': [], 'Running': [], 'Cycling': [], 'Swimming': []}
        self.intermediate_dict = {'Data_amount': [], 'Running': [], 'Cycling': [], 'Swimming': []}
        self.advance_dict = {'Data_amount': [], 'Running': [], 'Cycling': [], 'Swimming': []}
        self.athletes_dict = {}

    def plot_valid_TSS_pie(self, save=False):
        def is_valid_number(value):
            try:
                float(value)
                if float(value) != 0:
                    return True
                else:
                    return False
            except ValueError:
                return False

        tss_dict = {'TSS Valid': 0, 'TSS Calculable': 0, 'TSS Others': 0}
        file_names = ['Simon R Gronow (Novice).csv', 'Eduardo Oliveira (Intermediate).csv', 'Juliet Forsyth (Advance).csv',
                      # 'Michelle Bond (Advance).csv', 'Narelle Hayes (Novice).csv', 'Ollie Allan (Advance).csv',
                      # 'Rach Madden (High Intermediate).csv', # 'Sam Woodland (Advance World Champion).csv',
                      'Sophie Perry (Advance).csv']
        for file_name in file_names:
            athlete_dataframe = DataLoader(file_name).load_athlete_dataframe()
            preprocessor = DataPreprocessor(athlete_dataframe)
            total = preprocessor.athlete_dataframe.shape[0]
            valid = preprocessor.athlete_dataframe[
                preprocessor.athlete_dataframe['Training Stress Score®'].apply(lambda x: is_valid_number(x))].shape[0]
            activity_types = preprocessor.get_activity_types()
            for activity_type in activity_types:
                preprocessor.fill_out_tss(activity_type)
            calculable = preprocessor.athlete_dataframe[preprocessor.athlete_dataframe['Training Stress Score®']
                .apply(lambda x: is_valid_number(x))].shape[0] - valid

            tss_dict['TSS Valid'] += valid
            tss_dict['TSS Calculable'] += calculable
            tss_dict['TSS Others'] += total - valid - calculable

        tss_pie_labels = ['TSS Valid', 'TSS Calculable', 'TSS Others']
        tss_pie_sizes = [tss_dict['TSS Valid'], tss_dict['TSS Calculable'], tss_dict['TSS Others']]
        tss_pie_explode = (0, 0, 0.1)  # only "explode" the 3rd slice
        fig2, ax2 = plt.subplots()
        plt.rcParams["figure.figsize"] = (8, 5)
        ax2.pie(tss_pie_sizes, explode=tss_pie_explode, labels=tss_pie_labels, autopct='%1.1f%%',
                shadow=True, startangle=90, colors=('steelblue', 'skyblue', 'lightgrey'))
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        if save:
            plt.savefig('{}/plots/athlete_tss_pie.jpg'.format(os.path.pardir), format='jpg', dpi=1000)
        else:
            plt.show()

    def fill_out_dicts(self):
        def fill_out_level_dicts(dict, activity_counts):
            dict['Data_amount'].append(athlete_dataframe.shape[0])
            dict['Running'].append(activity_counts['Running'])
            dict['Cycling'].append(activity_counts['Cycling'])
            dict['Swimming'].append(activity_counts['Pool Swimming'])
        data_path = '{}/data'.format(os.path.pardir)
        dirs = os.listdir(data_path)
        for file_name in dirs:
            if file_name.endswith(".csv"):
                athlete_dataframe = DataLoader(file_name).load_athlete_dataframe()
                athlete_name = file_name.split(' ')[0]
                activity_counts = athlete_dataframe['Activity Type'].value_counts()
                self.athletes_dict[athlete_name] = {'Running': activity_counts['Running'],
                                               'Cycling': activity_counts['Cycling'],
                                               'Swimming': activity_counts['Pool Swimming']}
                for activity in ['Indoor Cycling', 'Road Cycling', 'Swimming', 'Open Water Swimming']:
                    try:
                        self.athletes_dict[athlete_name][activity.split(' ')[1]] += activity_counts[activity]
                    except:
                        pass
                if 'Novice' in file_name:
                    fill_out_level_dicts(self.novice_dict, activity_counts)
                if 'Intermediate' in file_name:
                    fill_out_level_dicts(self.intermediate_dict, activity_counts)
                if 'Advance' in file_name:
                    fill_out_level_dicts(self.advance_dict, activity_counts)


    def plot_activity_tendency_bar(self, save=False):
        self.fill_out_dicts()
        plt.rcParams["figure.figsize"] = (10, 5)
        pd.DataFrame(self.athletes_dict).T.plot(kind='bar', color=('steelblue', 'skyblue', 'lightgrey'))
        plt.xticks(rotation=30, ha='right')
        plt.title('Athlete Activity Tendencies')
        plt.ylabel('Activity Counts')
        if save:
            plt.savefig('{}/plots/athlete_activity_bar.jpg'.format(os.path.pardir), format='jpg', dpi=1200)
        else:
            plt.show()


    def plot_athlete_level_pie(self, save=False):
        self.fill_out_dicts()
        plt.rcParams["figure.figsize"] = (8, 5)
        pie_labels = ['Novice', 'Intermediate', 'Advance']
        sizes = [sum(self.novice_dict['Data_amount']),
                 sum(self.intermediate_dict['Data_amount']),
                 sum(self.advance_dict['Data_amount'])]
        explode = (0, 0, 0.1)  # only "explode" the 3rd slice
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=pie_labels, autopct='%1.1f%%',
                shadow=True, startangle=90, colors=('yellowgreen', 'olivedrab', 'forestgreen'))
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        if save:
            plt.savefig('{}/plots/athlete_athlete_level_pie.jpg'.format(os.path.pardir), format='jpg', dpi=1200)
        else:
            plt.show()

    def plot_boxplots(self, save=False):
        self.fill_out_dicts()
        fig, ax = plt.subplots()
        ax.set_title('Running Sample Sizes')
        ax.boxplot([self.novice_dict['Running'], self.intermediate_dict['Running'], self.advance_dict['Running']])
        plt.xticks([1, 2, 3], ['Novice', 'Intermediate', 'Advance'])
        plt.show()

        fig1, ax1 = plt.subplots()
        ax1.set_title('Cycling Sample Sizes')
        ax1.boxplot([self.novice_dict['Cycling'], self.intermediate_dict['Cycling'], self.advance_dict['Cycling']])
        plt.xticks([1, 2, 3], ['Novice', 'Intermediate', 'Advance'])
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.set_title('Swimming Sample Sizes')
        ax2.boxplot([self.novice_dict['Swimming'], self.intermediate_dict['Swimming'], self.advance_dict['Swimming']])
        plt.xticks([1, 2, 3], ['Novice', 'Intermediate', 'Advance'])
        plt.show()


    def plot_frequency(self, save=False):
        data_path = '{}/data'.format(os.path.pardir)
        dirs = os.listdir(data_path)
        for file_name in dirs:
            # if file_name.endswith(".csv"):
            if file_name == 'Andrea Stranna (High Intermediate).csv':
                athlete_dataframe = DataLoader(file_name).load_athlete_dataframe()
                dates = [date.split(' ')[0] for date in list(athlete_dataframe['Date'].values)]
                athlete_dataframe['Date'] = athlete_dataframe['Date'].str.split(' ').str[0]
                fig, ax = plt.subplots(figsize=(8, 4.5))
                df = athlete_dataframe[['Date', 'Activity Type']].groupby(['Date'])
                df.count()['Activity Type'].plot(ax=ax)
                fig.autofmt_xdate()
                plt.xticks(rotation=30, ha='right')
                plt.show()


if __name__ == '__main__':
    create_plot_folder()
    single_plotter = SingleAthleteDataPlotter('Simon R Gronow (Novice).csv')
    # single_plotter.plot_PMC()
    multi_plotter = MultipleAtheletesDataPlotter()
    multi_plotter.plot_valid_TSS_pie(save=False)
    multi_plotter.plot_athlete_level_pie(save=False)
    multi_plotter.plot_activity_tendency_bar(save=True)
    multi_plotter.plot_frequency(save=True)


