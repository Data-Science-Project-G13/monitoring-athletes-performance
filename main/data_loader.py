"""Athletes Data Loader

This script allows the user to load the athletes data.

This tool accepts comma separated value files (.csv).

This script requires that `pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * load_athlete_dataframe - returns a pandas data_frame
    * main - the main function of the script
"""

import os
import pandas as pd


class DataLoader():
    """
    A class used to load data

    ...

    Attributes
    ----------
    file_or_dir_name : str
        The name of the file that is about to load (default '{}/data')

    Methods
    -------
    load_athlete_dataframe()
        Load the data frame
    """

    def __init__(self, data_type='original'):
        """The constructor include the parameter data_type which indicates original/additional data
        so that it can avoid data structure problems given that the two functions in the class
        have return in different data types.

        """
        self.data_path = '{}/data'.format(os.path.pardir)
        self.data_type = data_type

    def load_original_data(self, file_name):
        """Load the original data for an athlete

        Returns
        -------
        Pandas data frame
        """
        if self.data_type == 'original':
            file_path = '{}/{}'.format(self.data_path, file_name)
            return pd.read_csv(file_path, sep=',')
        if self.data_type == 'additional':
            print('Invalid function call. Given type \'additional\'.')
            return None


    def load_additional_data(self, athletes_name, activity_type='', split_type=''):
        """Load the additional data for an athlete

        Parameters
        -------
        activity_type : str
            The activity type. Options are '', 'cycling', 'running', 'swimming' and 'training'/
        split_type : str
            The split and laps types. Options are '', 'real-time', 'laps', 'starts'
        athletes_name : str
            The name of the athlete from whom the data converted from fit files is about to get

        Returns
        -------
        Returns a list of file names in strings converted from fit files
        """
        if self.data_type == 'original':
            print('Invalid function call. Given type \'original\'.')
            return None
        if self.data_type == 'additional':
            dir_path = '{}/csv_{}/fit_csv'.format(self.data_path, '_'.join(athletes_name.lower().split()))
            return ['{}/{}'.format(dir_path, file_name) for file_name in os.listdir(dir_path)
                    if file_name.startswith(activity_type) and file_name.endswith('{}.csv'.format(split_type))]



if __name__ == '__main__':
    # Load original data
    data_loader_original = DataLoader()
    print(data_loader_original.load_original_data('Simon R Gronow (Novice).csv').head())

    # Load additional data
    data_loader_additional = DataLoader('additional')
    print(data_loader_additional.load_additional_data('eduardo oliveira'))
    print(data_loader_additional.load_additional_data('eduardo oliveira', 'swimming', 'real-time'))

