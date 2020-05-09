import os
import pandas as pd

class DataLoader():

    def __init__(self, file_name):
        data_path = '{}/data'.format(os.path.pardir)
        self.file_path = '{}/{}'.format(data_path, file_name)

    def load_athlete_dataframe(self):
        return pd.read_csv(self.file_path, sep=',')