"""Main Script of the Project

This script allows the user to run the project.

This file can also be imported as a module and contains the following
functions:

    * get_nonzero_TSS_rows - returns a pandas data_frame which contains rows with valid TSS
    * main - the main function of the script
"""

from data_loader import DataLoader
from data_preprocess import DataPreprocessor
from models import ModelBuilder


def get_nonzero_TSS_rows(athlete_df):
    valid_df = athlete_df.loc[athlete_df['Training Stress Score®'] != 0]
    return valid_df


def main(athlete_df):
    valid_df = get_nonzero_TSS_rows(athlete_df)
    y = valid_df[columns_for_analysis[0]]
    X = valid_df[columns_for_analysis[1:]]
    data_preprocessor = DataPreprocessor()
    data_preprocessor.clean_numerical_columns(X, columns_for_analysis[1:])
    model_builder = ModelBuilder(X, y)
    X_train, X_test, y_train, y_test = model_builder.split_train_validation()
    model_builder.process_linear_regression(X_train, y_train)


if __name__ == '__main__':
    file_name = 'Simon R Gronow (Novice).csv'
    columns_for_analysis = ['Training Stress Score®', 'Avg HR', 'Avg Power']
    data_loader = DataLoader('original')
    athlete_df = data_loader.load_original_data(file_name)
    main(athlete_df)


