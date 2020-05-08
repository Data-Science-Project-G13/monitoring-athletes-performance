import os
import pandas as pd
from models import ModelBuilder
from data_preprocess import DataPreprocessor


def load_athlete_df(file_name):
    data_path = '{}/data'.format(os.path.pardir)
    athlete_csv_file = '{}/{}'.format(data_path, file_name)
    athlete_df = pd.read_csv(athlete_csv_file)
    return athlete_df


def get_nonzero_TSS_rows(athlete_df):
    valid_df = athlete_df.loc[athlete_df['Training Stress Score®'] != 0]
    return valid_df


if __name__ == '__main__':
    file_name = 'Simon R Gronow - Ironman Build Cairns IM 2019 - Activities.csv'
    columns_for_analysis = ['Training Stress Score®', 'Avg HR', 'Avg Power']
    athlete_df = load_athlete_df(file_name)
    valid_df = get_nonzero_TSS_rows(athlete_df)
    y = valid_df[columns_for_analysis[0]]
    X = valid_df[columns_for_analysis[1:]]
    data_preprocessor = DataPreprocessor()
    data_preprocessor.clean_numerical_columns(X, columns_for_analysis[1:])
    model_builder = ModelBuilder(X, y)
    X_train, X_test, y_train, y_test = model_builder.split_train_validation()
    model_builder.process_linear_regression(X_train, y_train)


