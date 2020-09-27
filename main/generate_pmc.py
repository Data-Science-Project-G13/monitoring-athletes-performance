# Packages
import os
import pandas as pd
# Self-defined modules
import data_feature_engineering


def _add_fitness_fatigue(spreadsheet):
    spreadsheet['ATL'] = spreadsheet.rolling('7d', min_periods=1, on='Date')['Training Stress Score®'].mean()
    spreadsheet['CTL'] = spreadsheet.rolling('42d', min_periods=1, on='Date')['Training Stress Score®'].mean()


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


def process_pmc_generation():
    pass


if __name__ == '__main__':
    process_pmc_generation()