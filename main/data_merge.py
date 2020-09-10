import data_feature_engineering

def merge_spreadsheet_additional(athletes_name):
    """Merge the spreadsheet data and the additional data
    """
    spreadsheet = data_feature_engineering.main('spreadsheet', athletes_name)
    additionals = data_feature_engineering.main('additional', athletes_name)

    for add in additionals:
        pass
    for index, record in spreadsheet.iterrows():
        print(record['Date'].split(' ')[0])
        break


if __name__ == '__main__':
    athletes_names = ['eduardo oliveira']
    merge_spreadsheet_additional(athletes_names[0])
