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
import utility
from data_loader import DataLoader



class SpreadsheetDataFeatureExtractor():

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

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

    def __init__(self, file_name: str, athletes_css: float=None, athletes_lact_thr: (float,float)=None):
        self.file_name = file_name
        self.activity_type = self._get_activity_type()
        self.session_df = pd.read_csv(file_name)
        self.critical_swim_speed = athletes_css
        self.athletes_lact_thr = athletes_lact_thr
        session_datetime = self.file_name.split('_')[-3]
        self.features_extracted = {'Date': session_datetime, 'Activity Type': self.activity_type,
                                   'TSS': float(0), 'Other Feature 1': float(0), 'Other Feature 2': float(0)}

    def _get_activity_type(self):
        activity_types = ['running', 'cycling', 'swimming']
        for activity_type in activity_types:
            if activity_type in self.file_name:
                return activity_type
        return None

    def _get_tss_for_session_running(self):
        # TODO: Please complete the function. This is related to hrTSS.
        #  self.dataframe is the dataframe for the csv file.
        #  self.activity_type is the activity type.
        #  The function returns TSS (as a float).
        #  Return float(0) if can't compute. Please handle this situation.
        #  @Spoorthi @Sindhu
        jf_lact_thr = self.athletes_lact_thr[0]
        ac_lact_thr = self.athletes_lact_thr[1]
        if self.session_df['heart_rate'].isnull().values.any() == True:
            tss=float(0)
            return tss
        time, heart_rate = list(self.session_df['time_in_seconds']), list(self.session_df['heart_rate'])
        average_heart_rate = sum(heart_rate) / len(heart_rate)
        time_elapsed=time[len(time)-1]-time[0]
        if jf_lact_thr*0 <= average_heart_rate < jf_lact_thr*0.283 :
            tss=20*(time_elapsed/3600)
        if jf_lact_thr*0.283 <= average_heart_rate < jf_lact_thr*0.566:
            tss=30*(time_elapsed/3600)
        if jf_lact_thr*0.566 <= average_heart_rate < jf_lact_thr*0.85:
            tss=40*(time_elapsed/3600)
        if jf_lact_thr*0.85 <= average_heart_rate < jf_lact_thr*0.875:
            tss=50*(time_elapsed/3600)
        if jf_lact_thr*0.875 <= average_heart_rate <= jf_lact_thr*0.89:
            tss=60*(time_elapsed/3600)
        if jf_lact_thr*0.89 < average_heart_rate <= jf_lact_thr*0.94:
            tss=70*(time_elapsed/3600)
        if jf_lact_thr*0.94 < average_heart_rate <= jf_lact_thr*0.99:
            tss=80*(time_elapsed/3600)
        if jf_lact_thr*0.99 < average_heart_rate <= jf_lact_thr*1.02:
            tss=100*(time_elapsed/3600)
        if jf_lact_thr*1.02 < average_heart_rate <= jf_lact_thr*1.05:
            tss=120*(time_elapsed/3600)
            print(tss, "tss7888")
        if jf_lact_thr*1.05 < average_heart_rate:
            tss=140*(time_elapsed/3600)
        return tss

    def _get_tss_for_session_cycling(self):
        # TODO: Please complete the function. This is related to cycling TSS.
        #  self.dataframe is the dataframe for the csv file.
        #  self.activity_type is the activity type.
        #  The function returns TSS (as a float).
        #  Return float(0) if can't compute. Please handle this situation.
        #  @Spoorthi @Sindhu
        tss = float(0)
        return tss

    def _get_tss_for_session_swimming(self):
        # TODO: Please complete the function. This is related to swimming TSS.
        #  self.dataframe is the dataframe for the csv file.
        #  self.activity_type is the activity type.
        #  self.critical_swimming_speed is the critical swimming speed for the athlete.
        #  The function returns TSS (as a float).
        #  Return float(0) if can't compute. Please handle this situation.
        #  @Lin @Yuhan
        if self.critical_swim_speed:
            """ Implement function here @Lin """
            pass
        else:
            return float(0)

    def _extract_tss(self):
        tss = None
        if self.activity_type == 'running':
            tss = self._get_tss_for_session_running()
        elif self.activity_type == 'cycling':
            tss = self._get_tss_for_session_cycling()
        elif self.activity_type == 'swimming':
            tss = self._get_tss_for_session_swimming()
        return tss

    def _extract_other_feature_1(self):
        return None

    def _extract_other_feature_2(self):
        return None

    def _show_processing_info(self, state: str, verbose=False):
        """ Display the feature engineering processing information

        Parameters
        -------
        state: str
            The state of the process.
        verbose: bool
            If true, display the messages, not display otherwise.
        """
        if not verbose:
            return
        if state == 'start':
            print('\nProcessing feature engineering from {} ...'.format(self.file_name[3:]))
            if self.activity_type not in ['running', 'cycling', 'swimming']:
                print('Not able to process feature engineering for this activity type.')
        elif state == 'end':
            print('Feature engineering finished.')

    def process_feature_engineering(self):
        self._show_processing_info('start', verbose=False)
        self.features_extracted['TSS'] = self._extract_tss()
        self.features_extracted['Other Feature 1'] = self._extract_other_feature_1()
        self.features_extracted['Other Feature 2'] = self._extract_other_feature_2()
        self._show_processing_info('end', verbose=False)
        return self.features_extracted


def _function_for_testing(file_name, test_type):
    """
    """
    """*** This is the function that is used for testing only and will be removed ***"""
    if test_type in file_name: return True
    else: return False


def main(data_type: str, athletes_name: str):
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

    if data_type == 'spreadsheet':
        spreadsheet_feature_extractor = SpreadsheetDataFeatureExtractor(cleaned_spreadsheet_data_frame)
        spreadsheet_feature_extractor.process_feature_engineering()
        return spreadsheet_feature_extractor.dataframe

    elif data_type == 'additional':
        athletes_css = utility.get_athlete_css(athletes_name)
        athletes_lact_thr = utility.get_athletes_lact_thr(athletes_name)
        data_loader_additional = DataLoader('additional')
        cleaned_additional_data_filenames = data_loader_additional.load_cleaned_additional_data(athletes_name)
        if cleaned_additional_data_filenames:
            additional_features = {'features': ['Other Feature 1', 'Other Feature 2']}
            for file_name in cleaned_additional_data_filenames:
                # TODO: A reminder, you can use the function below to test your functions for ONE .csv file instead all
                #  For example if you are testing cycling TSS, just use the function below.
                #  If you want to test running, change the test_type to 'running'. Similarly for swimming.
                #  @Spoorthi @Sindhu @Lin @Yuhan
                test_type = 'running'
                if not _function_for_testing(file_name, test_type):
                    continue
                additional_feature_extractor = AdditionalDataFeatureExtractor(file_name,
                                                                              athletes_css=athletes_css,
                                                                              athletes_lact_thr=athletes_lact_thr)
                features_extracted = additional_feature_extractor.process_feature_engineering()
                additional_features[features_extracted['Date']] = features_extracted
                print('Preview of the features extracted: \n', features_extracted)
                # TODO: If you want to test all the files, comment out the '_function_for_testing's below.
                if _function_for_testing(file_name, test_type):
                    break
            return additional_features
        else:
            return None


if __name__ == '__main__':
    athletes_names = ['eduardo oliveira']
    main('spreadsheet', athletes_names[0])
    main('additional', athletes_names[0])

