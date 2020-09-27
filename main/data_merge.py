# Packages
import os
import pandas as pd
# Self-defined modules
import data_feature_engineering


def _save_merged_df(file_name, merged_dataframe: pd.DataFrame, verbose):
    merged_data_folder = '{}/data/merged_dataframes'.format(os.path.pardir)
    if not os.path.exists(merged_data_folder):
        os.mkdir(merged_data_folder)
    merged_dataframe.to_csv('{}/{}'.format(merged_data_folder, file_name), index=False)
    if verbose:
        print('Merged {} data saved!'.format(file_name))


def _add_rolling_tss(spreadsheet):
    # TODO: Use previous TSS rather than including current
    spreadsheet['ROLL TSS SUM'] = spreadsheet.rolling('14d', min_periods=1, on='Date')['Training Stress Score®'].sum()


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
            date = str(record['Date']).split(' ')[0]  # + record['Date'].split(' ')[1][:2]
            activity_type = record['Activity Type'].split(' ')[-1]
            if date in additionals.keys():
                # and activity_type.lower() == additionals[date]['Activity Type']:
                # Additional data activities and spreadsheet not always matched
                if additionals[date]['TSS'] is not None and additionals[date]['TSS'] != 0:
                    spreadsheet.at[index, 'Training Stress Score®'] = additionals[date]['TSS']
                for feature in additional_features:
                    spreadsheet.at[index, feature] = additionals[date][feature]
            # Replace TSS with None when it equals to zero for a more accurate ATL CTL calculation
            if spreadsheet.at[index, 'Training Stress Score®'] == 0:
                spreadsheet.at[index, 'Training Stress Score®'] = None
    spreadsheet['Date'] = pd.to_datetime(spreadsheet['Date'])
    _add_rolling_tss(spreadsheet)
    return spreadsheet


def process(athletes_name, verbose=False):
    merged_df = merge_spreadsheet_additional(athletes_name)
    file_name = 'merged_{}.csv'.format('_'.join(athletes_name.split(' ')))
    _save_merged_df(file_name, merged_df, verbose)


if __name__ == '__main__':
    athletes_name = ['Eduardo Oliveira']
    process(athletes_name[0])
