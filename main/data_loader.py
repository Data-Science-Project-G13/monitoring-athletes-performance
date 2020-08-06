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
    sample_str : str
        Description
    file_name : str
        The name of the file that is about to load (default '{}/data')

    Methods
    -------
    load_athlete_dataframe()
        Load the data frame
    """

    sample_str = ''
    def __init__(self, file_name):
        data_path = '{}/data'.format(os.path.pardir)
        self.file_path = '{}/{}'.format(data_path, file_name)

    def load_athlete_dataframe(self):
        """Load the CoachingMate data for the athlete

        Returns
        -------
        Pandas data frame
        """
        return pd.read_csv(self.file_path, sep=',')

    def load_athlete_garmin_dataframe(self):
        pass


if __name__ == '__main__':
    data_loader = DataLoader('Simon R Gronow (Novice).csv')
    print(data_loader.load_athlete_dataframe().head())
