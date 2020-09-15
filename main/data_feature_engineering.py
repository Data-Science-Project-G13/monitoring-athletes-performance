"""Data feature engineering

This script allows the user to do feature engineering on the athletes data.

This tool accepts comma separated value files (.csv).

This script requires that `pandas` be installed within the Python
environment you are running this script in.

"""
# Packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
# Self-defined modules
import utility
import data_merge
from data_loader import DataLoader


class SpreadsheetDataFeatureExtractor():

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def _add_activity_duration(self):
        def time_to_seconds(hh_mm_ss):
            hh, mm, ss = map(float, hh_mm_ss.split(':'))
            return hh*60*60 + mm*60 + ss
        self.dataframe['Duration'] = self.dataframe['Time'].apply(lambda x: time_to_seconds(x))

    def _add_activity_encode(self):
        # Encode the activities to five more general categories
        activities = self.dataframe['Activity Type'].astype('category').cat.categories.tolist()
        codes = []
        for activity in activities:
            if 'swimming' in activity.lower():
                codes.append(1)
            elif 'cycling' in activity.lower():
                codes.append(2)
            elif 'running' in activity.lower():
                codes.append(3)
            elif 'strength' in activity.lower():
                codes.append(4)
            else:
                codes.append(5)
        encode_dict = {k: v for k, v in zip(activities, codes)}
        self.dataframe['Activity Code'] = self.dataframe['Activity Type'].copy()
        self.dataframe['Activity Code'].replace(encode_dict, inplace=True)
        # Then use one-hot encoder for a better encoding
        encoder = LabelBinarizer()
        encoder.fit(self.dataframe['Activity Code'])
        transformed = encoder.transform(self.dataframe['Activity Code'])
        one_hot_df = pd.DataFrame(transformed,
                                  columns=['Activity Code 0', 'Activity Code 1', 'Activity Code 2', 'Activity Code 3', 'Activity Code 4'])
        self.dataframe = pd.concat([self.dataframe, one_hot_df], axis=1)

    def _add_num_activity_types_in_one_week(self):
        self.dataframe['Date'] = pd.to_datetime(self.dataframe['Date'])
        self.dataframe['Num Uniq Acts Weekly'] = self.dataframe.rolling('7d', min_periods=1, on='Date')['Activity Code']\
            .apply(lambda arr: pd.Series(arr).nunique())
        # print(list(self.dataframe['Num Activities']))

    def process_feature_engineering(self):
        """
        Process Feature Engineering on a spreadsheet data
        """
        self._add_activity_duration()
        self._add_activity_encode()
        self._add_num_activity_types_in_one_week()


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
        jf_lact_thr = self.athletes_lact_thr[0]
        ac_lact_thr = self.athletes_lact_thr[1]
        tss = float(0)
        if self.session_df['heart_rate'].isnull().values.any() == True:
            return tss
        time, heart_rate = list(self.session_df['time_in_seconds']), list(self.session_df['heart_rate'])
        average_heart_rate = sum(heart_rate) / len(heart_rate)
        time_elapsed=time[len(time)-1]-time[0]
        if jf_lact_thr*0 <= average_heart_rate < jf_lact_thr*0.283 :
            tss=20*(time_elapsed/3600)
        elif jf_lact_thr*0.283 <= average_heart_rate < jf_lact_thr*0.566:
            tss=30*(time_elapsed/3600)
        elif jf_lact_thr*0.566 <= average_heart_rate < jf_lact_thr*0.85:
            tss=40*(time_elapsed/3600)
        elif jf_lact_thr*0.85 <= average_heart_rate < jf_lact_thr*0.875:
            tss=50*(time_elapsed/3600)
        elif jf_lact_thr*0.875 <= average_heart_rate <= jf_lact_thr*0.89:
            tss=60*(time_elapsed/3600)
        elif jf_lact_thr*0.89 < average_heart_rate <= jf_lact_thr*0.94:
            tss=70*(time_elapsed/3600)
        elif jf_lact_thr*0.94 < average_heart_rate <= jf_lact_thr*0.99:
            tss=80*(time_elapsed/3600)
        elif jf_lact_thr*0.99 < average_heart_rate <= jf_lact_thr*1.02:
            tss=100*(time_elapsed/3600)
        elif jf_lact_thr*1.02 < average_heart_rate <= jf_lact_thr*1.05:
            tss=120*(time_elapsed/3600)
            # print(tss, "tss7888")
        elif jf_lact_thr*1.05 < average_heart_rate:
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
        if self.critical_swim_speed:
            CSS = self.critical_swim_speed
            df = self.session_df
            dist_list = []
            time_list = []
            for i in range(len(df) - 1):
                dist = df.distance[i + 1] - df.distance[i]
                time = df.time_in_seconds[i + 1] - df.time_in_seconds[i]
                if time < 0:
                    continue
                dist_list.append(dist)
                time_list.append(time)
            Distance, Duration = sum(dist_list) * 1000, sum(time_list)
            Pace = Duration / (Distance / 100)
            TSS = (CSS / Pace) ** 3 * (Duration / 3600) * 100
            return TSS
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


def _function_for_testing(file_name, test_type='all'):
    """
    """
    """*** This is the function that is used for testing only and will be removed ***"""
    if test_type == 'all': return True
    elif test_type in file_name: return True
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
                #  If you want to test running, change the test_type to 'running'. Similarly for swimming and cycling.
                test_type = 'all'
                if not _function_for_testing(file_name, test_type):
                    continue
                additional_feature_extractor = AdditionalDataFeatureExtractor(file_name,
                                                                              athletes_css=athletes_css,
                                                                              athletes_lact_thr=athletes_lact_thr)
                features_extracted = additional_feature_extractor.process_feature_engineering()
                additional_features[features_extracted['Date']] = features_extracted
                # print('Preview of the features extracted: \n', features_extracted)
            return additional_features
        else:
            return None


if __name__ == '__main__':
    athletes_names = ['eduardo oliveira']
    main('spreadsheet', athletes_names[0])
    # main('additional', athletes_names[0])

