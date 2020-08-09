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
    example_attribute : str
        Description
    file_or_dir_name : str
        The name of the file that is about to load (default '{}/data')

    Methods
    -------
    load_athlete_dataframe()
        Load the data frame
    """

    example_attribute = ''
    def __init__(self, file_or_dir_name, data_type='original'):
        self.data_path = '{}/data'.format(os.path.pardir)
        self.file_or_dir_name = file_or_dir_name
        self.data_type = data_type

    def load_original_data(self):
        """Load the original data for an athlete

        Returns
        -------
        Pandas data frame
        """
        if self.data_type == 'original':
            file_path = '{}/{}'.format(self.data_path, self.file_or_dir_name)
            return pd.read_csv(file_path, sep=',')
        if self.data_type == 'additional':
            print('Invalid function call. Given type \'additional\'.')
            return None


    def load_additional_data(self, activity_type='', split_type=''):
        """Load the additional data for an athlete

        Parameters
        -------
        activity_type : str
            Activity type. Options are '', 'cycling', 'running', 'swimming' and 'training'/
        split_type : str
            Split and laps types. Options are '', 'real_time', 'laps', 'starts'


        Returns : list(str)
        -------
        Returns a list of file names converted from fit files
        """
        if self.data_type == 'original':
            print('Invalid function call. Given type \'original\'.')
            return None
        if self.data_type == 'additional':
            dir_path = '{}/csv_{}/fit_csv'.format(self.data_path, self.file_or_dir_name)
            return [file_name for file_name in os.listdir(dir_path)
                    if file_name.startswith(activity_type) and file_name.endswith('{}.csv'.format(split_type))]



if __name__ == '__main__':
    # Load original data
    data_loader_original = DataLoader('Simon R Gronow (Novice).csv')
    print(data_loader_original.load_original_data().head())

    # Load additional data
    data_loader_additional = DataLoader('eduardo_oliveira', 'additional')
    print(data_loader_additional.load_additional_data())
    print(data_loader_additional.load_additional_data('swimming', 'laps'))

