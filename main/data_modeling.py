# Packages
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score
# Self-defined modules
import utility
import data_loader


# Set the data frame display option
pd.set_option('display.max_row', 20)
pd.set_option('display.max_columns', 20)


class TrainLoadModelBuilder():

    def __init__(self, dataframe):
        features = ['Distance', 'Duration', 'Calories', 'Avg HR', 'Max HR', 'Avg Speed', 'Max Speed',
                    'Training Stress Score®',
                    'Normalized Power® (NP®)', 'Max Avg Power (20 min)', 'Avg Power', 'Max Power']
        self.X = dataframe.loc[:, dataframe.columns != 'TSS']
        self.y = dataframe['Training Load Indicator']
        self.random_state = np.random.RandomState(0)

    def _split_train_validation(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.25, random_state = 25)
        return X_train, X_test, y_train, y_test

    def _process_linear_regression(self, X_train, y_train):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        # print('Linear Regression Coefficients', '')
        return lr

    def _process_svm_classification(self, X_train, y_train):
        classifier = svm.LinearSVC(random_state=self.random_state)
        classifier.fit(X_train, y_train)
        return classifier

    def _validate_model(self, X_test, y_test, classifier: svm):
        accuracy, precision, recall, f1_score = 0, 0, 0, 0
        y_score = classifier.decision_function(X_test)
        n_classes = 3
        precision, recall, average_precision = dict(), dict(), dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
        return precision, recall, average_precision

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        classifier = self._process_svm_classification(X_train, y_train)
        precision, recall, average_precision = self._validate_model(X_test, y_test, classifier)


class PerformanceModelBuilder():

    def __init__(self):
        pass


def process_train_load_modeling(athletes_name):
    loader = data_loader.DataLoader()
    data_set = loader.load_merged_data(athletes_name)
    data_set_modeling = data_set[data_set['Training Stress Score®'].notnull()]
    print(data_set_modeling)


def process_performance_modeling(athletes_name):
    pass


if __name__ == '__main__':
    athletes_names = ['eduardo oliveira']
    process_train_load_modeling(athletes_names[0])
    process_performance_modeling(athletes_names[0])