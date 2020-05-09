import os
import pandas as pd


class DataPreprocessor():

    def __init__(self):
        pass

    def check_valid_tss_perc(self):
        data_path = '{}/data'.format(os.path.pardir)
        dirs = os.listdir(data_path)
        for file_name in dirs:
            # if file.endswith((".csv", ".xlsx")):
            if file_name.endswith(".csv"):
                csv_file = '{}/{}'.format(data_path, file_name)
                data = pd.read_csv(csv_file)
                tss_col = data['Training Stress Score®']
                tss_non_zero_perc = sum([1 for val in tss_col if float(str(val).replace(',','')) != 0])/len(tss_col)
                print('{} TSS Non zero percentage: {}'
                      .format(file_name.split('.')[0], tss_non_zero_perc))

    def view_data(self, file_name):
        data_path = '{}/data'.format(os.path.pardir)
        csv_file = '{}/{}'.format(data_path, file_name)
        data = pd.read_csv(csv_file)
        tss_col = data['Training Stress Score®']
        print(tss_col)
        return tss_col

    def get_valid_data(self):
        '''
        Get rows that contains TSS or contains enough values for calculating TSS
        :return:
        '''
        pass

    def clean_numerical_columns(self, df, columns):
        pass

    def clean_categorical_columns(self, df, columns):
        pass


if __name__ == '__main__':
    data_preprocessor = DataPreprocessor()
    data_preprocessor.check_valid_tss_perc()
    file_name = 'Andrea Stranna Activities Jul18 to Apr20.csv'
    tss_col = data_preprocessor.view_data(file_name)

