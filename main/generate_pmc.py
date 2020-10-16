# Packages
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
# Self-defined modules
import utility
from data_loader import DataLoader


def _add_fitness_fatigue(dataframe):
    dataframe['Date'] = pd.to_datetime(dataframe.Date)
    dataframe['Fatigue (ATL)'] = dataframe.rolling('7d', min_periods=1, on='Date')['Training Stress Score®'].mean()
    dataframe['Fitness (CTL)'] = dataframe.rolling('42d', min_periods=1, on='Date')['Training Stress Score®'].mean()


def _add_form(dataframe):
    dataframe['Form (TSB)'] = dataframe['Fitness (CTL)'] - dataframe['Fatigue (ATL)']


def _label_data_record(spreadsheet):
    """Label the data for modeling
    Label 1 if over-training, 0 if appropriate, 1 if under-training
    """
    indicators = []
    for index, record in spreadsheet.iterrows():
        form = record['CTL'] - record['ATL']
        if form >= 5:
            indicators.append(0)
        elif form <= -30:
            indicators.append(2)
        else:
            indicators.append(1)
    spreadsheet['Training Load Indicator'] = pd.Series(indicators, index=spreadsheet.index)


def _generate_pmc(athletes_name, dataframe, display_tss=True, save=True):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    dates = dataframe['Date']
    TSS, ATL, CTL, TSB = 'Training Stress Score®', 'Fatigue (ATL)', 'Fitness (CTL)', 'Form (TSB)'

    def plot_TSS():
        tss = dataframe[TSS]
        ax.plot(dates, tss, '.', label='TSS', color='lightskyblue')
        ax.legend(loc=2)
        # rotate and align the tick labels so they look better
        fig.autofmt_xdate()
        ax.yaxis.set_label_position("left")
        # ax.set_xlabel('Dates')
        ax.set_ylabel('TSS')
        ax.set_ylim([50, 250])

    def plot_fatigue_and_fitness():
        ax2.plot(dates, dataframe[ATL], label=ATL, color='green')
        ax2.plot(dates, dataframe[CTL], label=CTL, color='orange')
        ax2.plot(dates, dataframe[TSB], label=TSB, color='pink')
        ax2.set_ylabel('CTL / ATL / TSB')
        ax2.set_ylim([-100, 275])

    def plot_zones():
        year, month, day = str(dates.iloc[0]).split(' ')[0].split('-')
        end_date = datetime.date(int(year),int(month), int(day))
        font_size = 9
        line_params = [(-30, 'red'), (-10, 'green'), (5, 'grey'), (25, 'royalblue')]
        text_params = [(-42, 'red', 'High Risk Zone'), (-22, 'green', 'Optimal Training Zone'),
                       (-7, 'grey', 'Grey Zone'), (13, 'royalblue', 'Freshness Zone'),
                       (30, 'darkgoldenrod', 'Transition Zone')]
        for params in line_params:
            plt.axhline(xmin=0.05, xmax=1, y=params[0], color=params[1], linestyle='-')
        for params in text_params:
            plt.text(x=end_date, y=params[0], s=params[2], horizontalalignment='right',
                     color=params[1], fontsize=font_size)

    if display_tss:
        plot_TSS()
    plot_fatigue_and_fitness()
    plot_zones()
    plt.title('Performance Management Chart - {}'.format(athletes_name.title()))
    plt.legend()
    plt.savefig('{}/plots/PMC - {}.jpg'.format(os.path.pardir, athletes_name.title()),
                format='jpg', dpi=1200) if save else plt.show()


def get_tss_estimated_data(athletes_name):
    TSS = 'Training Stress Score®'
    original_merged_data = DataLoader().load_merged_data(athletes_name=athletes_name)
    null_tss_data = original_merged_data[~original_merged_data[TSS].notnull()]
    sub_dataframe_dict = utility.split_dataframe_by_activities(original_merged_data)
    model_types = utility.get_train_load_model_types(athletes_name)
    for activity, sub_dataframe in sub_dataframe_dict.items():
        best_model_type_for_activity = model_types[activity]
        if best_model_type_for_activity:
            sub_dataframe_for_modeling = sub_dataframe[sub_dataframe[TSS].notnull()]
            regressor = utility.load_model(athletes_name, activity, best_model_type_for_activity)
            general_features = utility.FeatureManager().get_common_features_among_activities()
            activity_specific_features = utility.FeatureManager().get_activity_specific_features(activity)
            features = [feature for feature in general_features + activity_specific_features
                        if feature in sub_dataframe_for_modeling.columns and feature != TSS
                        and not sub_dataframe[feature].isnull().any()]
            X, y = sub_dataframe[features], list(sub_dataframe[TSS])
            y_pred = regressor.predict(X)
            y_final = [y[i] if y[i] is np.nan else y_pred[i] for i in range(len(y))]
            original_merged_data.loc[sub_dataframe.index, TSS] = y_final
        else:
            pass
            # print("No model for {}'s {} activity".format(athletes_name.title(), activity))
    return original_merged_data


def process_pmc_generation(athletes_name, display_tss=True, save_pmc_figure=True):
    dataframe_for_pmc = get_tss_estimated_data(athletes_name)
    _add_fitness_fatigue(dataframe_for_pmc)
    _add_form(dataframe_for_pmc)
    _generate_pmc(athletes_name, dataframe_for_pmc, display_tss=display_tss, save=save_pmc_figure)


if __name__ == '__main__':
    athletes_names = ['eduardo oliveira', 'xu chen', 'carly hart']
    for athletes_name in athletes_names:
        process_pmc_generation(athletes_name, display_tss=True)