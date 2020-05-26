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


pd.set_option('display.max_row', 20)
pd.set_option('display.max_columns', 4)


class DataPlotter():

    def __init__(self, athlete_dataframe):
        self.athlete_dataframe =  athlete_dataframe
        self.fig, self.ax = plt.subplots()
        # self.fig2, self.ax2 = plt.subplots()
        self.ax2 = self.ax.twinx()

    def plot_TSS(self):
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
        plt.text(x=datetime.date(2019, 8, 1), y=-40, s='High Risk Zone', horizontalalignment='right', color='red')
        plt.text(x=datetime.date(2019, 8, 1), y=-20, s='Optimal Training Zone', horizontalalignment='right', color='green')
        plt.text(x=datetime.date(2019, 8, 1), y=-4, s='Grey Zone', horizontalalignment='right', color='grey')
        plt.text(x=datetime.date(2019, 8, 1), y=15, s='Freshness Zone', horizontalalignment='right', color='royalblue')
        plt.text(x=datetime.date(2019, 8, 1), y=32, s='Transition Zone', horizontalalignment='right', color='darkgoldenrod')
        self.ax2.set_ylabel('CTL / ATL / TSB')


if __name__ == '__main__':
    # data_path = '{}/data'.format(os.path.pardir)
    # dirs = os.listdir(data_path)
    # for file_name in dirs:
    #     if file_name.endswith(".csv"):
    #     if file_name == 'Ollie Allan (Advance).csv':
    #         athlete_dataframe = DataLoader(file_name).load_athlete_dataframe()
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
    file_names = ['Simon R Gronow (Novice).csv']  # 'Eduardo Oliveira (Intermediate).csv'
    for file_name in file_names:
        athlete_dataframe = DataLoader(file_name).load_athlete_dataframe()
        preprocessor = DataPreprocessor(athlete_dataframe)
        total = preprocessor.athlete_dataframe.shape[0]
        valid = preprocessor.athlete_dataframe[preprocessor.athlete_dataframe['Training Stress Score®'].apply(lambda x: is_valid_number(x))].shape[0]
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
    ax2.pie(tss_pie_sizes, explode=tss_pie_explode, labels=tss_pie_labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=('steelblue', 'skyblue', 'lightgrey'))
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    # plt.savefig('{}/plots/athlete_tss_pie.jpg'.format(os.path.pardir), format='jpg', dpi=1000)

        # preprocessor = DataPreprocessor(athlete_dataframe)
        # activity_types = preprocessor.get_activity_types()
        # for activity_type in activity_types:
        #     preprocessor.fill_out_tss(activity_type)
        # df = preprocessor.athlete_dataframe
        # df['Date'] = pd.to_datetime(df.Date, dayfirst=True)
        # df = df.sort_values(by=['Date'], ascending=True)
        # plotter = DataPlotter(df)
        # plotter.plot_TSS()
        # plotter.plot_fatigue_and_fitness()
        # plt.title('Performance Management Chart - {}'.format(file_name.split('.')[0]))
        # plt.legend()
        # # plt.show()
        # plt.savefig('{}/plots/PMC - {}.jpg'.format(os.path.pardir, file_name.split('.')[0]), format='jpg', dpi=1200)
