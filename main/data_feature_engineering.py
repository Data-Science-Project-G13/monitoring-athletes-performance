"""Data feature engineering

This script allows the user to do feature engineering on the athletes data.

This tool accepts comma separated value files (.csv).

This script requires that `pandas` be installed within the Python
environment you are running this script in.

"""
# Packages
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import sklearn.decomposition as dp

# Self-defined modules
from data_loader import DataLoader



class OriginalDataFeatureExtractor():

    def __init__(self):
        pass

    def process_pca(self, low_dimension, path):
        """
        PCA (Principal Component Analysis)
        A method used to dimension reduction, selecting the most effective variables
        Parameters
        -------
        low_dimension : int
            How many variables left after PCA processing
        file_name : str
            Path of raw dataset needs to be reduced
        """
        data = np.loadtxt(path, dtype=str, delimiter=',')
        # read athlete dataset
        col_name = data[0,]
        # variables name of the dataset
        pca = PCA(n_components=low_dimension)
        # set how many variables will be kept
        data_processed = pca.fit_transform(data[1:, ])
        # use PCA function to calculate the effect (from 0 to 1) of each variables
        variance_ratio = dict(zip(col_name, pca.explained_variance_ratio_))
        rank = sorted(variance_ratio.items(), key=lambda x: x[1], reverse=True)
        # variables' effect are ranked in descend order
        return rank


class AdditionalDataFeatureExtractor():

    def __init__(self, file_name):
        self.file_name = file_name

    def get_tss_for_session(self):
        pass


    def process_feature_engineering(self):
        print('\nProcessing feature engineering from {} ...'.format(self.file_name[3:]))
        if 'running' in self.file_name:
            athlete_df = pd.DataFrame(pd.read_csv(self.file_name))
        elif 'cycling' in self.file_name:
            athlete_df = pd.DataFrame(pd.read_csv(self.file_name))
        elif 'swimming' in self.file_name:
            athlete_df = pd.DataFrame(pd.read_csv(self.file_name))
        else:
            print('Not able to process feature engineering for this activity type.')


def main(data_type: str, athletes_name: str) :
    """The main function of processing feature engineering

    Parameters
    -------
    data_type: str
       The type of the data, spreadsheet or additional.
    athletes_name: str
        The name of the athlete whose data is about to clean.
    """

    if data_type == 'spreadsheet' :
        pass

    elif data_type == 'additional' :
        data_loader_additional = DataLoader('additional')
        cleaned_additional_data_filenames = data_loader_additional.load_cleaned_additional_data(athletes_name)
        for file_name in cleaned_additional_data_filenames:
            additional_feature_extractor = AdditionalDataFeatureExtractor(file_name)
            additional_feature_extractor.process_feature_engineering()


if __name__ == '__main__':
    athletes_names = ['eduardo oliveira']
    main('additional', athletes_names[0])
