import os
import pandas as pd
import data_feature_engineering

pd.set_option('display.max_row', 200)
pd.set_option('display.max_columns', 10)

def _save_merged_df(file_name, merged_dataframe: pd.DataFrame):
    merged_data_folder = '{}/data/merged_dataframes'.format(os.path.pardir)
    if not os.path.exists(merged_data_folder):
        os.mkdir(merged_data_folder)
    merged_dataframe.to_csv('{}/{}'.format(merged_data_folder, file_name))
    print('Merged {} data saved!'.format(file_name))


def merge_spreadsheet_additional(athletes_name):
    """Merge the spreadsheet data and the additional data
    """
    spreadsheet = data_feature_engineering.main('spreadsheet', athletes_name)
    additionals = data_feature_engineering.main('additional', athletes_name)

    for index, record in spreadsheet.iterrows():
        date = record['Date'].split(' ')[0]  # + record['Date'].split(' ')[1][:2]
        if date in additionals.keys():
            spreadsheet.at[index, 'Training Stress Score®'] = additionals[date]['TSS']
    return spreadsheet


def main():
    athletes_names = ['eduardo oliveira']
    merged_df = merge_spreadsheet_additional(athletes_names[0])
    file_name = 'merged_{}'.format('_'.join(athletes_names[0].split(' ')))
    _save_merged_df(file_name, merged_df)

if __name__ == '__main__':
    main()
