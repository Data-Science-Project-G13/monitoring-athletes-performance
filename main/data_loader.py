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


def check_valid_tss_perc():
    """Gets and prints the spreadsheet's header columns

    Parameters
    ----------
    file_loc : str
        The file location of the spreadsheet
    print_cols : bool, optional
        A flag used to print the columns to the console (default is
        False)

    Returns
    -------
    list
        a list of strings used that are the header columns
    """
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
        return pd.read_csv(self.file_path, sep=',')


if __name__ == '__main__':
    check_valid_tss_perc()
