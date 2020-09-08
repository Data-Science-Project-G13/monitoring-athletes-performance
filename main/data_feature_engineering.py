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

# Self-defined modules
from data_loader import DataLoader



class SpreadsheetDataFeatureExtractor():

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def fill_out_tss(self, tss_list: []):
        # TODO: Tingli will implement this
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

    def process_feature_engineering(self):
        """
        Process Feature Engineering on a spreadsheet data
        """
        """ *** Main function of the extractor @Spoorthi @Sindhu, the self.dataframe 
        should be done feature engineering after call this function*** """
        pass


class AdditionalDataFeatureExtractor():

    def __init__(self, file_name: str):
        self.file_name = file_name
        self.activity_type = self._get_activity_type()
        self.session_df = pd.read_csv(file_name)
        # TODO: get swimming speed for the athlete from the lower layer
        self.critical_swimming_speed = None

    def _get_activity_type(self):
        activity_types = ['running', 'cycling', 'swimming']
        for activity_type in activity_types:
            if activity_type in self.file_name:
                return activity_type
        return None

    def get_tss_for_session_running(self):
        # TODO: Please complete the function. This is related to hrTSS.
        #  self.dataframe is the dataframe for the csv file.
        #  self.activity_type is the activity type.
        #  @Spoorthi @Sindhu
        pass

    def get_tss_for_session_cycling(self):
        # TODO: Please complete the function. This is related to cycling TSS.
        #  self.dataframe is the dataframe for the csv file.
        #  self.activity_type is the activity type.
        #  @Spoorthi @Sindhu
        pass

    def get_tss_for_session_swimming(self):
        # TODO: Please complete the function. This is related to swimming TSS.
        #  self.dataframe is the dataframe for the csv file.
        #  self.activity_type is the activity type.
        #  self.critical_swimming_speed is the critical swimming speed for the athlete.
        #  @Spoorthi @Sindhu
        pass

    def process_feature_engineering(self):
        print('\nProcessing feature engineering from {} ...'.format(self.file_name[3:]))
        if self.activity_type == 'running':
            pass
        elif self.activity_type == 'cycling':
            pass
        elif self.activity_type == 'swimming':
            pass
        else:
            print('Not able to process feature engineering for this activity type.')


def _match_dates_activities_spreadsheet_additional(cleaned_spreadsheet_data_frame):
    spread_sheet_dates = cleaned_spreadsheet_data_frame['Date']
    index = 0
    return index


def main(data_type: str, athletes_name: str) :
    """The main function of processing feature engineering

    Parameters
    -------
    data_type: str
       The type of the data, spreadsheet or additional.
    athletes_name: str
        The name of the athlete whose data is about to clean.
    """
    data_loader_spreadsheet = DataLoader('spreadsheet')
    cleaned_spreadsheet_data_frame = data_loader_spreadsheet.load_cleaned_spreadsheet_data(athletes_name=athletes_name)

    if data_type == 'spreadsheet' :
        spreadsheet_feature_extractor = SpreadsheetDataFeatureExtractor(cleaned_spreadsheet_data_frame)

    elif data_type == 'additional' :
        data_loader_additional = DataLoader('additional')
        cleaned_additional_data_filenames = data_loader_additional.load_cleaned_additional_data(athletes_name)
        for file_name in cleaned_additional_data_filenames:
            additional_feature_extractor = AdditionalDataFeatureExtractor(file_name)
            additional_feature_extractor.process_feature_engineering()


if __name__ == '__main__':
    athletes_names = ['eduardo oliveira']
    main('spreadsheet', athletes_names[0])
    main('additional', athletes_names[0])
