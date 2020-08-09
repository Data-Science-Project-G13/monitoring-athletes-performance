import os
import pandas as pd
import math
import datetime as dt
from sklearn.linear_model import LinearRegression
from data_loader import DataLoader


pd.set_option('display.max_row', 100)
pd.set_option('display.max_columns', 50)
data_integrity_baseline = 0.8
minimum_num_rows_for_calculate_trimp_tss_coef = 30


class DataPreprocessor():
    """
    A class used to preprocess the data so that the one can make PMC from it

    ...

    Attributes
    ----------
    athlete_dataframe : pandas data frame
        The name of the file that is about to load (default '{}/data')
    reg : linear regression model
        The TRIMP-TSS regression model for the athlete

    Methods
    -------
    get_activity_types()
        Load the data frame
    add_time_in_minutes()
    calculate_rtss()
        Calculate running TSS
    calculate_stss()
        Calculate swimming TSS
    calculate_hrtss()
        Calculate heart rate TSS
    calculate_trimp_exp()
        Calculate TRIMP
    calculate_ttss()
        Calculate TRIMP TSS
    fill_out_tss()
        Fill out the missing TSS of the column
    """

    def __init__(self, athlete_dataframe):
        self.athlete_dataframe = athlete_dataframe
        self.reg = self._get_trimp_tss_reg_for_this_athlete()


    def view_data(self):
        print(self.athlete_dataframe)

    def get_activity_types(self):
        activity_types = self.athlete_dataframe['Activity Type'].unique()
        return activity_types

    def add_time_in_minutes(self, activity_df):
        total_minutes_list = []
        for time in activity_df['Time']:
            h_m_s = time.split(':')
            total_seconds = 0
            for t in h_m_s:
                t = t.replace(',', '.')
                total_seconds = total_seconds * 60 + float(t)
            total_minutes_list.append(total_seconds / 60)
        activity_df.insert(3, 'Time in Minutes', total_minutes_list, True)
        return activity_df

    def calculate_rtss(self, activity_df):
        return None

    def calculate_stss(self, activity_df):
        return None

    def calculate_hrtss(self, activity_df):
        return None

    def calculate_trimp_exp(self, activity_df, gender='male'):
        if gender == 'male': y = 1.92
        else: y = 1.67
        time_duration = activity_df['Time in Minutes']
        avg_hr = activity_df['Avg HR']
        max_hr = activity_df['Max HR']
        hr_rest = 60
        hrr = (avg_hr - hr_rest) / (max_hr - hr_rest)
        trimp = time_duration*hrr*0.64*math.exp(y*hrr)
        return trimp

    def _get_trimp_tss_reg_for_this_athlete(self):
        activity_types = self.get_activity_types()
        for activity_type in activity_types:
            activity_df = self.athlete_dataframe[self.athlete_dataframe['Activity Type'] == activity_type]
            num_nonzeros = activity_df.fillna(0).astype(bool).sum(axis=0)['Training Stress Score®']
            nonzero_perc = num_nonzeros/activity_df.shape[0]
            if nonzero_perc > data_integrity_baseline:
                print('Activity type {} has TSS {}% filled, with number of rows {}.'
                      .format(activity_type, round(nonzero_perc*100, 2), activity_df.shape[0]))
                if activity_df.shape[0] > minimum_num_rows_for_calculate_trimp_tss_coef:
                    activity_df = self.add_time_in_minutes(activity_df)
                    activity_df = activity_df[activity_df['Avg HR'] != '--']
                    activity_df['Avg HR'] = activity_df['Avg HR'].astype(float)
                    activity_df['Max HR'] = activity_df['Max HR'].astype(float)
                    trimp = activity_df.apply(self.calculate_trimp_exp, axis=1)
                    tss = pd.DataFrame(activity_df['Training Stress Score®'])
                    reg = LinearRegression(fit_intercept=True).fit(pd.DataFrame(trimp), tss)
                    return reg

    def calculate_ttss(self, activity_df):
        try:
            activity_df = self.add_time_in_minutes(activity_df)
            activity_df_new = activity_df[(activity_df['Avg HR'] != '--') & (activity_df['Max HR'] != '--')].copy()
            if activity_df_new.shape[0] > 0:
                activity_df_new['Avg HR'] = activity_df_new['Avg HR'].astype(float)
                activity_df_new['Max HR'] = activity_df_new['Max HR'].astype(float)
                trimp = activity_df_new.apply(self.calculate_trimp_exp, axis=1)
                ttss = pd.DataFrame(self.reg.predict(pd.DataFrame(trimp)), columns=['Training Stress Score®'])
                return ttss
        except Exception as e:
            print(e)
            # raise e

    def fill_out_tss(self, activity_type):
        activity_df = self.athlete_dataframe[self.athlete_dataframe['Activity Type'] == activity_type]
        num_nonzeros = activity_df.fillna(0).astype(bool).sum(axis=0)['Training Stress Score®']
        rtss_requirements_satisfied = False
        stss_requirements_satisfied = False
        hrtss_requirements_satisfied = False
        ttss_requirements_satisfied = True
        # Check if the number of TSS values is big enough
        if num_nonzeros/activity_df.shape[0] > data_integrity_baseline:
            # May use Interpolation for only few values missing
            tss = activity_df['Training Stress Score®']
            tss = pd.DataFrame({'Training Stress Score®': [float(str(var).replace(',', '')) for var in tss]})
        elif activity_type == 'Running' and rtss_requirements_satisfied:
            tss = self.calculate_rtss(activity_df)
        elif activity_type == 'Swimming' and stss_requirements_satisfied:
            tss = self.calculate_stss(activity_df)
        elif hrtss_requirements_satisfied:
            tss = self.calculate_hrtss(activity_df)
        elif ttss_requirements_satisfied:
            tss = self.calculate_ttss(activity_df)
            if tss is not None:
                # Update TSS with estimated ones. Important.
                self.athlete_dataframe.loc[(self.athlete_dataframe['Activity Type'] == activity_type)
                                           & (self.athlete_dataframe['Avg HR'] != '--')
                                           & (self.athlete_dataframe['Max HR'] != '--'),
                                           'Training Stress Score®'] = tss['Training Stress Score®'].values
        else:
            tss = None

    def get_fatigue(self):
        pass

    def get_fitness(self):
        pass

    def get_form(self):
        pass

    def get_valid_data(self):
        """Get rows that contains TSS or contains enough values for calculating TSS
        """
        pass

    def clean_numerical_columns(self, df, columns):
        pass

    def clean_categorical_columns(self, df, columns):
        pass


if __name__ == '__main__':
    file_name = 'Ollie Allan (Advance).csv'
    athlete_dataframe = DataLoader().load_original_data(file_name)
    preprocessor = DataPreprocessor(athlete_dataframe)
    # preprocessor.view_data()
    activity_types = preprocessor.get_activity_types()
    for activity_type in activity_types:
        preprocessor.fill_out_tss(activity_type)
    print(preprocessor.athlete_dataframe)




