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


pd.set_option('display.max_row', 10)
pd.set_option('display.max_columns', 4)


class DataPlotter():

    def __init__(self, athlete_dataframe):
        self.athlete_dataframe =  athlete_dataframe

    def plot_TSS(self):
        df = self.athlete_dataframe
        dates = df['Date'] #mdates.num2date(mdates.datestr2num(df['Date']))
        tss = df['Training Stress Score®']
        tss = pd.DataFrame([float(str(var).replace(',', '')) for var in tss])
        fig, ax = plt.subplots()
        ax.plot(dates, tss, '.', label='TSS')
        # rotate and align the tick labels so they look better
        fig.autofmt_xdate()

    def plot_fatigue(self):
        df = self.athlete_dataframe
        dates = df['Date']
        tss = df['Training Stress Score®']
        tss = pd.DataFrame([float(str(var).replace(',', '')) for var in tss])
        atl = tss.rolling(7, min_periods=1).mean()
        plt.plot(dates, atl, label='Fatigue')

    def plot_fitness(self):
        df = self.athlete_dataframe
        dates = df['Date']
        tss = df['Training Stress Score®']
        tss = pd.DataFrame([float(str(var).replace(',', '')) for var in tss])
        ctl = tss.rolling(42, min_periods=1).mean()
        plt.plot(dates, ctl, label='Fitness')


if __name__ == '__main__':
    # data_path = '{}/data'.format(os.path.pardir)
    # dirs = os.listdir(data_path)
    # for file_name in dirs:
    #     # if file_name.endswith(".csv"):
    #     if file_name == 'Ollie Allan (Advance).csv':
    #         athlete_dataframe = DataLoader(file_name).load_athlete_dataframe()
    #         plotter = DataPlotter(athlete_dataframe)
    #         plotter.plot_TSS()
    file_name = 'Simon R Gronow (Novice).csv'
    athlete_dataframe = DataLoader(file_name).load_athlete_dataframe()
    preprocessor = DataPreprocessor(athlete_dataframe)
    activity_types = preprocessor.get_activity_types()
    for activity_type in activity_types:
        preprocessor.fill_out_tss(activity_type)
    df = preprocessor.athlete_dataframe
    df['Date'] = pd.to_datetime(df.Date, dayfirst=True)
    df = df.sort_values(by=['Date'], ascending=True)
    plotter = DataPlotter(df)
    plotter.plot_TSS()
    plotter.plot_fatigue()
    plotter.plot_fitness()
    plt.title('Performance Management Chart {}'.format(file_name))
    plt.ylabel('CTL / ATL / TSB')
    plt.legend()
    plt.show()
