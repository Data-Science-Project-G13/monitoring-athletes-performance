"""Data feature engineering

This script allows the user to do feature engineering on the athletes data.

This tool accepts comma separated value files (.csv).

This script requires that `pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * load_athlete_dataframe - returns a pandas data_frame
    * main - the main function of the script
"""

import featuretools
import pandas
import numpy
import sklearn
import csv
import numpy as np
from sklearn.decomposition import PCA
import sklearn.decomposition as dp
from sklearn.datasets.base import load_iris





class OriginalDataFeatureExtractor():

    def __init__(self):
        pass


class AdditionalDataFeatureExtractor():

    def __init__(self):
        pass

    def get_tss_for_session(self):
        pass

def PCA_Process(low_dimension,path):
    """
    PCA(Principal Component Analysis)
    A method used to dimension reduction, selecting the most effective variables

    Parameters
    -------
    path : str
    Path of raw dataset needs to be reduced
    low_dimension : int
    How many variables left after PCA processing

    """
    data = np.loadtxt(path,dtype=str,delimiter=',')
    # read athlete dataset
    col_name = data[0,]
    # variables name of the dataset
    pca = PCA(n_components=low_dimension)
    # set how many variables will be kept
    data_processed = pca.fit_transform(data[1:,])
    # use PCA function to calculate the effect (from 0 to 1) of each variables
    variance_ratio = dict(zip(col_name,pca.explained_variance_ratio_))
    rank = sorted(variance_ratio.items(), key=lambda x: x[1], reverse=True)
    # variables' effect are ranked in descend order
    return rank





if __name__ == '__main__':




    pass

