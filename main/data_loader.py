import os
import pandas as pd


def check_valid_tss_perc():
    data_path = '{}/data'.format(os.path.pardir)
    dirs = os.listdir(data_path)
    for file_name in dirs:
        # if file.endswith((".csv", ".xlsx")):
        if file_name.endswith(".csv"):
            csv_file = '{}/{}'.format(data_path, file_name)
            data = pd.read_csv(csv_file)
            tss_col = data['Training Stress Score®']
            tss_non_zero_perc = sum([1 for val in tss_col if float(str(val).replace(',', '')) != 0]) / len(tss_col)
            print('{} TSS Non zero percentage: {}'
                  .format(file_name.split('.')[0], tss_non_zero_perc))


class DataLoader():

    def __init__(self, file_name):
        data_path = '{}/data'.format(os.path.pardir)
        self.file_path = '{}/{}'.format(data_path, file_name)

    def load_athlete_dataframe(self):
        return pd.read_csv(self.file_path, sep=',')


if __name__ == '__main__':
    check_valid_tss_perc()
