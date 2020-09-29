# Packages
import os
import numpy as np
import pandas as pd
# Self-defined modules
import utility
from data_loader import DataLoader


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


def get_tss_estimated_data(athletes_name):
    original_merged_data = DataLoader().load_merged_data(athletes_name=athletes_name)
    null_tss_data = original_merged_data[~original_merged_data['Training Stress Score®'].notnull()]
    sub_dataframe_dict = utility.split_dataframe_by_activities(original_merged_data)

    for activity, sub_dataframe in sub_dataframe_dict.items():
        sub_dataframe_for_modeling = sub_dataframe[sub_dataframe['Training Stress Score®'].notnull()]
        best_model_type = 'random_forest'
        regressor = utility.load_model(athletes_name, activity, best_model_type)
        TSS = 'Training Stress Score®'
        general_features = utility.FeatureManager().get_common_features_among_activities()
        activity_specific_features = utility.FeatureManager().get_activity_specific_features(activity)

        features = [feature for feature in general_features + activity_specific_features
                    if feature in sub_dataframe_for_modeling.columns and feature != TSS
                    and not sub_dataframe[feature].isnull().any()]

        X = sub_dataframe[features]
        y = list(sub_dataframe[TSS])  # fill out only null cells and keep those not null
        y_pred = regressor.predict(X)
        y_final = [y[i] if y[i] is np.nan else y_pred[i] for i in range(len(y))]
        original_merged_data.loc[sub_dataframe.index, TSS] = y_final


def process_pmc_generation(athletes_name):
    data = get_tss_estimated_data(athletes_name)


if __name__ == '__main__':
    athletes_name = 'xu chen'
    process_pmc_generation(athletes_name)