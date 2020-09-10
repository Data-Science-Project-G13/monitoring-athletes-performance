import pandas as pd
import data_feature_engineering

pd.set_option('display.max_row', 200)
pd.set_option('display.max_columns', 10)

def merge_spreadsheet_additional(athletes_name):
    """Merge the spreadsheet data and the additional data
    """
    spreadsheet = data_feature_engineering.main('spreadsheet', athletes_name)
    additionals = data_feature_engineering.main('additional', athletes_name)

    for index, record in spreadsheet.iterrows():
        date = record['Date'].split(' ')[0]
        if date in additionals.keys():
            spreadsheet.at[index, 'Training Stress Score®'] = additionals[date]['TSS']
    print(list(spreadsheet['Training Stress Score®']))


if __name__ == '__main__':
    athletes_names = ['eduardo oliveira']
    merge_spreadsheet_additional(athletes_names[0])
