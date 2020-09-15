# Packages
import os
import pandas as pd
# Self-defined modules
import data_feature_engineering


def _save_merged_df(file_name, merged_dataframe: pd.DataFrame):
    merged_data_folder = '{}/data/merged_dataframes'.format(os.path.pardir)
    if not os.path.exists(merged_data_folder):
        os.mkdir(merged_data_folder)
    merged_dataframe.to_csv('{}/{}'.format(merged_data_folder, file_name), index=False)
    print('Merged {} data saved!'.format(file_name))


def _add_fitness_fatigue(spreadsheet):
    spreadsheet['Date'] = pd.to_datetime(spreadsheet['Date'])
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
            indicators.append(-1)
        elif form <= -30:
            indicators.append(1)
        else:
            indicators.append(0)
    spreadsheet['Training Load Indicator'] = pd.Series(indicators, index=spreadsheet.index)


def merge_spreadsheet_additional(athletes_name):
    """Merge the spreadsheet data and the features extracted from additional data
    """
    spreadsheet = data_feature_engineering.main('spreadsheet', athletes_name)
    additionals = data_feature_engineering.main('additional', athletes_name)
    if additionals:
        additional_features = additionals['features']
        for feature in additional_features:
            spreadsheet[feature] = pd.Series(0, index=spreadsheet.index)
        for index, record in spreadsheet.iterrows():
            date = record['Date'].split(' ')[0]  # + record['Date'].split(' ')[1][:2]
            activity_type = record['Activity Type'].split(' ')[-1]

            if date in additionals.keys():
                # and activity_type.lower() == additionals[date]['Activity Type']:
                # dditional data activities and spreadsheet not always matched
                if additionals[date]['TSS'] is not None and additionals[date]['TSS'] != 0:
                    spreadsheet.at[index, 'Training Stress Score®'] = additionals[date]['TSS']
                for feature in additional_features:
                    spreadsheet.at[index, feature] = additionals[date][feature]
            # Replace TSS with None when it equals to zero for a more accurate ATL CTL calculation
            if spreadsheet.at[index, 'Training Stress Score®'] == 0:
                spreadsheet.at[index, 'Training Stress Score®'] = None
    _add_fitness_fatigue(spreadsheet)
    _label_data_record(spreadsheet)
    print(spreadsheet.head())
    print(spreadsheet.columns)
    return spreadsheet


def main():
    athletes_names = ['eduardo oliveira']
    merged_df = merge_spreadsheet_additional(athletes_names[0])
    file_name = 'merged_{}.csv'.format('_'.join(athletes_names[0].split(' ')))
    _save_merged_df(file_name, merged_df)

if __name__ == '__main__':
    main()
