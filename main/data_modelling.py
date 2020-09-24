# Packages
import numpy as np
import pandas as pd
# from tensorflow.keras import Sequential, layers
# from tensorflow.keras.utils import to_categorical

from sklearn import svm
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.classifier import StackingCVClassifier
from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
# Self-defined modules
import utility
import data_loader



# Set the data frame display option
pd.set_option('display.max_row', 20)
pd.set_option('display.max_columns', 20)


class TrainLoadModelBuilder():

    def __init__(self, dataframe):
        features = ['Distance', 'Calories', 'Avg HR', 'Max HR',
                    'Training Stress Score®', 'Num Uniq Acts Weekly', 'Duration', 'ROLL TSS SUM',
                    'Activity Code 0', 'Activity Code 1', 'Activity Code 2', 'Activity Code 3', 'Activity Code 4',
                    'Normalized Power® (NP®)', 'Max Avg Power (20 min)', 'Avg Power', 'Max Power']
        self.num_features = len(features)
        self.X = dataframe[features]
        self.y = dataframe['Training Load Indicator']

    def _split_train_validation(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 25)
        return X_train, X_test, y_train, y_test

    # def _process_neural_network(self, X_train, X_test, y_train, y_test):
    #     neural_network = Sequential()
    #     verbose, epochs, batch_size = 0, 30, 4
    #     neural_network.add(layers.Dense(265, input_shape=(self.num_features,), activation='relu'))
    #     neural_network.add(layers.BatchNormalization())
    #     neural_network.add(layers.Dense(64, activation='relu'))
    #     neural_network.add(layers.Dense(1, activation='sigmoid'))
    #     neural_network.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    #     neural_network.fit(X_train, y_train,  validation_data=(X_test, y_test),
    #                        epochs=epochs, batch_size=batch_size, shuffle=True)
    #     # TODO: Add Grid Search
    #     train_acc = neural_network.evaluate(X_train, y_train, verbose=0)[1]
    #     test_acc = neural_network.evaluate(X_test, y_test, verbose=0)[1]
    #     print('Training Set Accuracy: {}, Test Set Accuracy: {}'.format(train_acc, test_acc))
    #     return neural_network

    def _process_random_forest(self, X_train, y_train):
        rfc = RandomForestClassifier(max_depth=3, random_state=0)
        rfc.fit(X_train, y_train)
        return rfc

    def _process_svm_classification(self, X_train, y_train):
        classifier = svm.SVC(C=1.0,kernel='poly',degree=5)
        #classifier = svm.SVC(class_weight='balanced')
        classifier.fit(X_train, y_train)
        return classifier
    def _process_xgboost(self, X_train, y_train):
        xgb = XGBClassifier(alpha=15, colsample_bytree=0.1,learning_rate=1, max_depth=5,reg_lambda=10.0)
        xgb.fit(X_train, y_train)
        return xgb

    def _process_stacking(self, X_train, y_train):
        knn = KNeighborsClassifier(n_neighbors=1)
        rf = RandomForestClassifier(max_depth=3,max_features =6,n_estimators=50,random_state=0)
        SVM = svm.SVC(C=1.0,kernel='poly',degree=5)
        Xgb = XGBClassifier(alpha=15, colsample_bytree=0.1,learning_rate=1, max_depth=5,reg_lambda=10.0)
        gnb = GaussianNB()
        lr = LogisticRegression(C = 10.0, dual=False,max_iter=100,solver='lbfgs')

        sclf = StackingCVClassifier(classifiers=[knn, rf,lr,SVM,Xgb],
                                    meta_classifier=gnb,
                                    random_state=42)
        sclf.fit(X_train,y_train)
        return sclf

        # params = {'kneighborsclassifier__n_neighbors': [1, 5],
        #           'randomforestclassifier__n_estimators': [10, 50],
        #           'randomforestclassifier__max_depth':[3, 5, 10, 13],
        #           'randomforestclassifier__max_features': [2, 4, 6, 8, 10],
        #           'XGBClassifier__n_estimators': [400,1000],
        #           # 'XGBClassifier__max_depth': [15,20,25],
        #           # 'XGBClassifier__reg_alpha': [1.1, 1.2, 1.3],
        #           # 'XGBClassifier__reg_lambda': [1.1, 1.2, 1.3],
        #           # 'XGBClassifier__subsample': [0.7, 0.8, 0.9],
        #           'meta_classifier__C' : [0.1, 10.0]}

        # grid = GridSearchCV(estimator=sclf,
        #                     param_grid=params,
        #                     cv=5,
        #                     refit=True)
        # grid.fit(X_train, y_train)

        # cv_keys = ('mean_test_score', 'std_test_score', 'params')
        #
        # for r, _ in enumerate(grid.cv_results_['mean_test_score']) :
        #     print("%0.3f +/- %0.2f %r"
        #           % (grid.cv_results_[cv_keys[0]][r],
        #              grid.cv_results_[cv_keys[1]][r] / 2.0,
        #              grid.cv_results_[cv_keys[2]][r]))
        #
        # print('Best parameters: %s' % grid.best_params_)
        # print('Accuracy: %.2f' % grid.best_score_)
        # return grid

    def _process_Adaboost(self, X_train, y_train):
        AB = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
        AB.fit(X_train,y_train)
        return AB

    def _validate_model(self, X_test, y_test, classifier):
        y_preds = classifier.predict(X_test)  # predict classes for y test
        print(y_preds)
        accuracy = accuracy_score(y_test, y_preds)
        precision = precision_score(y_test, y_preds, average='macro', zero_division=0)
        recall = recall_score(y_test, y_preds, average='macro')
        f1 = f1_score(y_test, y_preds, average='macro')
        # auc = roc_auc_score(y_test, y_pred_probs)
        # y_score = classifier.decision_function(X_test)
        # n_classes = 3
        # precision, recall, average_precision = dict(), dict(), dict()
        # for i in range(n_classes):
        #     precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        #     average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
        return accuracy, precision, recall, f1

    def _display_performance_results(self, model_name, accuracy, precision, recall, f1):
        print('Classifier: {}'.format(model_name))
        print('Accuracy: {}, Precision: {}, Recall: {}, F1 score: {}'
              .format(round(accuracy, 2), round(precision, 2), round(recall, 2), round(f1, 2)))

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        print('X-y train-test shapes', X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        print(np.unique(y_train,return_counts=True))
        # ============ Random Forest ============
        classifier = self._process_random_forest(X_train, y_train)
        accuracy, precision, recall, f1 = self._validate_model(X_test, y_test, classifier)
        self._display_performance_results('Random Forest', accuracy, precision, recall, f1)
        # ============ SVM ============
        classifier = self._process_svm_classification(X_train, y_train)
        accuracy, precision, recall, f1 = self._validate_model(X_test, y_test, classifier)
        self._display_performance_results('SVM', accuracy, precision, recall, f1)
        # ============ Neural Network ============
        # classifier = self._process_neural_network(X_train, X_test, y_train, y_test)
        # ============ xgboost ============
        classifier = self._process_xgboost(X_train, y_train)
        accuracy, precision, recall, f1 = self._validate_model(X_test, y_test, classifier)
        self._display_performance_results('xgboost', accuracy, precision, recall, f1)
        # ============ stacking ============
        classifier = self._process_stacking(X_train, y_train)
        accuracy, precision, recall, f1 = self._validate_model(X_test, y_test, classifier)
        self._display_performance_results('stacking', accuracy, precision, recall, f1)
        # ============ Adaboost ============
        classifier = self._process_Adaboost(X_train, y_train)
        accuracy, precision, recall, f1 = self._validate_model(X_test, y_test, classifier)
        self._display_performance_results('Adaboost', accuracy, precision, recall, f1)



class PerformanceModelBuilder():

    def __init__(self):
        pass


def process_train_load_modeling(athletes_name):
    loader = data_loader.DataLoader()
    data_set = loader.load_merged_data(athletes_name=athletes_name)
    data_set_modeling = data_set[data_set['Training Stress Score®'].notnull()]
    train_load_builder = TrainLoadModelBuilder(data_set_modeling)
    train_load_builder.process_modeling()


def process_performance_modeling(athletes_name):
    pass


if __name__ == '__main__':
    athletes_names = ['eduardo oliveira']
    process_train_load_modeling(athletes_names[0])
    process_performance_modeling(athletes_names[0])
