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

    def plot_valid_TSS_pie(self):
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
        ax2.pie(tss_pie_sizes, explode=tss_pie_explode, labels=tss_pie_labels, autopct='%1.1f%%',
                shadow=True, startangle=90, colors=('steelblue', 'skyblue', 'lightgrey'))
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()
        # plt.savefig('{}/plots/athlete_tss_pie.jpg'.format(os.path.pardir), format='jpg', dpi=1000)


if __name__ == '__main__':
    data_path = '{}/data'.format(os.path.pardir)
    dirs = os.listdir(data_path)
    for file_name in dirs:
        if file_name.endswith(".csv"):
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
            plotter.plot_fatigue_and_fitness()
            plt.title('Performance Management Chart - {}'.format(file_name.split('.')[0]))
            plt.legend()
            # plt.show()
            plt.savefig('{}/plots/PMC - {}.jpg'.format(os.path.pardir, file_name.split('.')[0]), format='jpg', dpi=1200)


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

    tss_pie_labels = ['TSS Valid', 'TSS Calculable', 'Others']
    tss_pie_sizes = [tss_dict['TSS Valid'], tss_dict['TSS Calculable'], tss_dict['TSS Others']]
    tss_pie_explode = (0, 0, 0.1)  # only "explode" the 3rd slice
    plt.rcParams["figure.figsize"] = (7, 5)
    fig2, ax2 = plt.subplots()
    ax2.pie(tss_pie_sizes, explode=tss_pie_explode, labels=tss_pie_labels, autopct='%1.1f%%',
            shadow=True, startangle=90, colors=('steelblue', 'skyblue', 'lightgrey'))
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    # plt.savefig('{}/plots/athlete_tss_pie.jpg'.format(os.path.pardir), format='jpg', dpi=1000)


    # novice_dict = {'Data_amount': [], 'Running': [], 'Cycling': [], 'Swimming': []}
    # intermediate_dict = {'Data_amount': [], 'Running': [], 'Cycling': [], 'Swimming': []}
    # advance_dict = {'Data_amount': [], 'Running': [], 'Cycling': [], 'Swimming': []}
    # athletes_dict = {}
    # def fill_out_dicts(dict, activity_counts):
    #     dict['Data_amount'].append(athlete_dataframe.shape[0])
    #     dict['Running'].append(activity_counts['Running'])
    #     dict['Cycling'].append(activity_counts['Cycling'])
    #     dict['Swimming'].append(activity_counts['Pool Swimming'])
    #
    # data_path = '{}/data'.format(os.path.pardir)
    # dirs = os.listdir(data_path)
    # for file_name in dirs:
    #     if file_name.endswith(".csv"):
    #         athlete_dataframe = DataLoader(file_name).load_athlete_dataframe()
    #         athlete_name = file_name.split(' ')[0]
    #         activity_counts = athlete_dataframe['Activity Type'].value_counts()
    #         athletes_dict[athlete_name] = {'Running': activity_counts['Running'],
    #                                        'Cycling': activity_counts['Cycling'],
    #                                        'Swimming': activity_counts['Pool Swimming']}
    #         try:
    #             athletes_dict[athlete_name]['Cycling'] += activity_counts['Indoor Cycling']
    #         except:
    #             pass
    #         try:
    #             athletes_dict[athlete_name]['Cycling'] += activity_counts['Road Cycling']
    #         except:
    #             pass
    #         try:
    #             athletes_dict[athlete_name]['Swimming'] += activity_counts['Swimming']
    #         except:
    #             pass
    #         try:
    #             athletes_dict[athlete_name]['Swimming'] += activity_counts['Open Water Swimming']
    #         except:
    #             pass
    #
    #         if 'Novice' in file_name:
    #             fill_out_dicts(novice_dict, activity_counts)
    #         if 'Intermediate' in file_name:
    #             fill_out_dicts(intermediate_dict, activity_counts)
    #         if 'Advance' in file_name:
    #            fill_out_dicts(advance_dict, activity_counts)
    #
    # plt.rcParams["figure.figsize"] = (10, 5)
    # pd.DataFrame(athletes_dict).T.plot(kind='bar', color=('steelblue', 'skyblue', 'lightgrey'))
    # plt.xticks(rotation=30, ha='right')
    # plt.title('Athlete Activity Tendencies')
    # plt.ylabel('Activity Counts')
    # # plt.show()
    # plt.savefig('{}/plots/athlete_activity_bar.jpg'.format(os.path.pardir), format='jpg', dpi=1200)


    # pie_labels = ['Novice', 'Intermediate', 'Advance']
    # sizes = [sum(novice_dict['Data_amount']),
    #          sum(intermediate_dict['Data_amount']),
    #          sum(advance_dict['Data_amount'])]
    # explode = (0, 0, 0.1)  # only "explode" the 3rd slice
    # fig1, ax1 = plt.subplots()
    # ax1.pie(sizes, explode=explode, labels=pie_labels, autopct='%1.1f%%',
    #         shadow=True, startangle=90)
    # ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # plt.show()


    # fig, ax = plt.subplots()
    # ax.set_title('Running Sample Sizes')
    # ax.boxplot([novice_dict['Running'], intermediate_dict['Running'], advance_dict['Running']])
    # plt.xticks([1, 2, 3], ['Novice', 'Intermediate', 'Advance'])
    # plt.show()
    #
    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Cycling Sample Sizes')
    # ax1.boxplot([novice_dict['Cycling'], intermediate_dict['Cycling'], advance_dict['Cycling']])
    # plt.xticks([1, 2, 3], ['Novice', 'Intermediate', 'Advance'])
    # plt.show()
    #
    # fig2, ax2 = plt.subplots()
    # ax2.set_title('Swimming Sample Sizes')
    # ax2.boxplot([novice_dict['Swimming'], intermediate_dict['Swimming'], advance_dict['Swimming']])
    # plt.xticks([1, 2, 3], ['Novice', 'Intermediate', 'Advance'])
    # plt.show()

            # athlete_dataframe = athlete_dataframe[athlete_dataframe['Training Stress Score®'].apply(lambda x: str(x).replace(',',''))]
            # tss_num_nonzeros = athlete_dataframe[athlete_dataframe['Training Stress Score®'].astype(float) != 0].shape[0]
            # tss_nonzero_perc = tss_num_nonzeros / athlete_dataframe.shape[0]
            # valid_tss_proportions.append(tss_nonzero_perc)

            # hr_num_nonzeros = athlete_dataframe.fillna(0).astype(bool).sum(axis=0)['Avg HR']
            # hr_nonzero_perc = hr_num_nonzeros / athlete_dataframe.shape[0]
            # valid_tss_proportions.append(hr_nonzero_perc)


    # print(valid_tss_proportions)
    # print(valid_hr_proportions)

