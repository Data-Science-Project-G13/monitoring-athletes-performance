# Packages
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras import Sequential, layers
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.utils import to_categorical
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from xgboost import XGBClassifier
import xgboost as xgb
from mlxtend.classifier import StackingCVClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from warnings import filterwarnings
import numpy as np
filterwarnings('ignore')

# Self-defined modules
import utility
import data_loader


# Set the data frame display option
pd.set_option('display.max_row', 20)
pd.set_option('display.max_columns', 20)


class TrainLoadModelBuilder():

    def __init__(self, dataframe, activity_features):
        TSS = 'Training Stress Score®'
        features = [feature for feature in activity_features
                    if feature != TSS
                    and not dataframe[feature].isnull().any()]
        print('Features used for modeling: ', features)
        self.num_features = len(features)
        self.X = dataframe[features]
        self.y = dataframe[TSS]

    def _split_train_validation(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.2, random_state = 25)
        print('Shapes:  X_train: {}, y_train: {}, X_test: {}, y_test: {}'
            .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
        return X_train, X_test, y_train, y_test

    def _validate_model_regression(self, X_test, y_test, learner):
        y_preds = learner.predict(X_test)  # predict classes for y test
        print('Predictions Overview: ', y_preds)
        mae = mean_absolute_error(y_test, y_preds)
        rmse = np.sqrt(mean_squared_error(y_test, y_preds))
        rsquared=r2_score(y_test, y_preds)
        return mae, rmse,rsquared

    def _display_performance_results_regression(self, model_name, mae, rmse, rsquared):
        print('Regressor: {}'.format(model_name))
        print('Mean Absolute Error: {}, Root Mean Squared Error: {}, R-squared: {}'
              .format(round(mae, 3), round(rmse, 3), round(rsquared, 3)))

    def _display_performance_results_classification(self, model_name, accuracy, precision, recall, f1):
        print('Classifier: {}'.format(model_name))
        print('Accuracy: {}, Precision: {}, Recall: {}, F1 score: {}'
              .format(round(accuracy, 2), round(precision, 2), round(recall, 2), round(f1, 2)))

    def _validate_model_classification(self, X_test, y_test, learner):
        y_preds = learner.predict(X_test)  # predict classes for y test
        accuracy = accuracy_score(y_test, y_preds)
        precision = precision_score(y_test, y_preds, average='macro', zero_division=0)
        recall = recall_score(y_test, y_preds, average='macro')
        f1 = f1_score(y_test, y_preds, average='macro')
        return accuracy, precision, recall, f1


class ModelLinearRegression(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        #print(list(X_train))
        #print(X_train.isnull().values.any())
        #print(np.isnan(X_train.values.any()))
        #print(X_train)
        regressor=LinearRegression()
        ###This works
        #regressor = Ridge(alpha=0.04, normalize=True)
        ####
        #print(min(y_train),max(y_train))
        #regressor=Lasso(alpha=0.04, normalize=True)
        regressor.fit(X_train, y_train)
        #scores = cross_val_score(regressor, X_train, y_train, cv=3)
        #print("Cross - validated scores:", scores)
        #################################
        # lm = LinearRegression()
        # lm.fit(X_train, y_train)
        # rfe = RFE(lm, n_features_to_select=4)
        # rfe = rfe.fit(X_train, y_train)
        ##############################
        return regressor
        #return rfe

    def process_modeling(self):
        # TODO: Spoorthi
        X_train, X_test, y_train, y_test = self._split_train_validation()
        ##########
        sfs = SFS(Lasso(alpha=0.04, normalize=True),
                  k_features=(3,6),
                  forward=True,
                  floating=False,
                  scoring='r2',
                  cv=3)
        sfs1 = sfs.fit(X_train, y_train)
        feat_cols = list(sfs1.k_feature_names_)
        print(feat_cols)
        regressor = self._build_model(X_train[feat_cols], y_train)
        mae, rmse, rsquared = self._validate_model_regression(X_test[feat_cols], y_test, regressor)
        ##########
        # regressor = self._build_model(X_train, y_train)
        # mae, rmse, rsquared = self._validate_model_regression(X_test, y_test, regressor)
        ##########
        self._display_performance_results_regression('Linear Regression', mae, rmse,rsquared)

        return rmse, regressor


class ModelSVM(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        classifier = svm.SVC(C=5, kernel='linear')
        classifier.fit(X_train, y_train)
        return classifier

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        classifier = self._build_model(X_train, y_train)
        mae, rmse, rsquared = self._validate_model_regression(X_test, y_test, classifier)
        self._display_performance_results_regression('SVM', mae, rmse, rsquared)


class ModelNeuralNetwork(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, X_test, y_train, y_test):
        X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        print(X_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        verbose, epochs, batch_size = 0, 30, 4
        neural_network = Sequential()
        neural_network.add(layers.Bidirectional(layers.LSTM(units = 32, input_shape = (X_train.shape[1], 1), return_sequences=True)))
        neural_network.add(layers.Dense(256, activation='relu'))
        neural_network.add(layers.Dropout(0.2))
        neural_network.add(layers.BatchNormalization())
        neural_network.add(layers.Dense(64, activation='relu'))
        neural_network.add(layers.Dense(1, kernel_initializer='normal', activation='linear'))
        neural_network.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
        neural_network.fit(X_train, y_train,  validation_data=(X_test, y_test),
                           epochs=epochs, batch_size=batch_size, shuffle=True)
        print('Prediction Overview: ', np.reshape(neural_network.predict(X_test), (X_test.shape[0], X_test.shape[1])))
        train_mae = neural_network.evaluate(X_train, y_train, verbose=0)[1]
        test_mae = neural_network.evaluate(X_test, y_test, verbose=0)[1]
        print('Training Set MAE: {}, Test Set MAE: {}'.format(train_mae, test_mae))
        return test_mae, neural_network

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        mae, neural_network = self._build_model(X_train, X_test, y_train, y_test)
        return mae, neural_network


class ModelRandomForest(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
#       rfc = RandomForestRegressor(max_depth=2, random_state=0)
        param_grid = {
            'bootstrap': [True],
            'max_depth': [6, 8, 10],
            'max_features': ['sqrt', 'log2'],
            'min_samples_leaf': [2, 3, 5],
            'min_samples_split': [2, 3],
            'n_estimators': [100, 500]}
        rf = RandomForestRegressor()
#       rfc = DecisionTreeRegressor(max_depth=5, min_weight_fraction_leaf= 1e-5, min_impurity_decrease = 1, min_samples_split=2)
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 3, n_jobs = -1, verbose=0)
#       rfc = RandomForestRegressor(max_depth=10,max_features=10,n_estimators=50,random_state=0)
        grid_search.fit(X_train, y_train)
        best_grid = grid_search.best_estimator_
        print(best_grid)
        return best_grid

    def process_modeling(self):
        # TODO: @Lin
        X_train, X_test, y_train, y_test = self._split_train_validation()
        regressor = self._build_model(X_train, y_train)
        mae, rmse, rsquared = self._validate_model_regression(X_test, y_test, regressor)
        self._display_performance_results_regression('Random Forest', mae, rmse,rsquared)
#       self._display_performance_results_regression('Decision Tree', mae, rmse,rsquared)

        return rmse, regressor


class TestModel(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        rfc = RandomForestRegressor(max_depth=2, random_state=0)
        rfc.fit(X_train, y_train)
        return rfc

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        regressor = self._build_model(X_train, y_train)
        mae, rmse, rsquared = self._validate_model_regression(X_test, y_test, regressor)
        return rmse, regressor


class ModelXGBoost(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        # TODO: Sindhu
        xgb_reg = xgb.XGBRegressor(learning_rate=0.01,colsample_bytree = 0.4,subsample = 0.2,n_estimators=1000,reg_alpha = 0.3,
                                     max_depth=3,min_child_weight=3,gamma=0,objective ='reg:squarederror')
        xgb_reg.fit(X_train, y_train)
        # scores = cross_val_score(xgb_reg, X_train, y_train, cv=4)
        # print("Cross - validated scores:", scores)
        return xgb_reg


    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        regressor = self._build_model(X_train, y_train)
        mae, rmse,rsquared = self._validate_model_regression(X_test, y_test, regressor)
        self._display_performance_results_regression('XGBoost', mae, rmse,rsquared)
        return rmse, regressor


class ModelStacking(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        knn = KNeighborsClassifier(n_neighbors=1)
        rf = RandomForestClassifier(max_depth=3,max_features=6,n_estimators=50,random_state=0)
        SVM = svm.SVC(C=1.0,kernel='poly',degree=5)
        Xgb = XGBClassifier(alpha=15, colsample_bytree=0.1,learning_rate=1, max_depth=5,reg_lambda=10.0)
        gnb = GaussianNB()
        lr = LogisticRegression(C = 10.0, dual=False, max_iter=100, solver='lbfgs')
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

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        regressor = self._build_model(X_train, y_train)
        mae, rmse,rsquared = self._validate_model_regression(X_test, y_test, regressor)
        self._display_performance_results_regression('XGBoost', mae, rmse,rsquared)
        return regressor


class ModelAdaBoost(TrainLoadModelBuilder):

    def __init__(self, dataframe, activity_features):
        super().__init__(dataframe, activity_features)

    def _build_model(self, X_train, y_train):
        # TODO: @Yuhan
        # the focus of parameter tuning are n_estimators and learning_rate
        param_grid = [
            {'base_estimator': [None], 'n_estimators':np.linspace(100,2000,20)},
            {'learning_rate': np.linspace(0.1,1,10), 'algorithm': ['SAMME.R'], 'random_state': [None]},
        ]

        # AB = AdaBoostClassifier(base_estimator=None, n_estimators=500, learning_rate=1.0,
        #                         algorithm='SAMME.R',  random_state=None)
        AB = AdaBoostClassifier()
        grid_search = GridSearchCV(AB, param_grid, cv=2,
                                   scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train.astype('int'))
        # added.astype('int')) in case of "Unknown label type: 'continuous'" happens
        return grid_search.best_estimator_

    def process_modeling(self):
        X_train, X_test, y_train, y_test = self._split_train_validation()
        regressor = self._build_model(X_train, y_train)
        mae, rmse,rsquared = self._validate_model_regression(X_test, y_test, regressor)
        self._display_performance_results_regression('AdaBoost', mae, rmse,rsquared)
        return rmse, regressor


class PerformanceModelBuilder():

    def __init__(self):
        pass


def process_train_load_modeling(athletes_name):
    loader = data_loader.DataLoader()
    data_set = loader.load_merged_data(athletes_name=athletes_name)
    sub_dataframe_dict = utility.split_dataframe_by_activities(data_set)
    best_model_dict = {}
    for activity, sub_dataframe in sub_dataframe_dict.items():
        utility.SystemReminder().display_activity_modeling_start(activity)
        sub_dataframe_for_modeling = sub_dataframe[sub_dataframe['Training Stress Score®'].notnull()]
        if sub_dataframe_for_modeling.shape[0] > 10:
            general_features = utility.FeatureManager().get_common_features_among_activities()
            activity_specific_features = utility.FeatureManager().get_activity_specific_features(activity)
            features = [feature for feature in general_features + activity_specific_features
                        if feature in sub_dataframe.columns
                        and not sub_dataframe[feature].isnull().any()]   # Handle columns with null

            def select_best_model():
                min_mae, best_model_type, best_regressor = float('inf'), '', None
                for model_class in [ModelRandomForest, ModelXGBoost, ModelAdaBoost]:
                    model_type = model_class.__name__[5:]
                    print('\nBuilding {}...'.format(model_type))
                    builder = model_class(sub_dataframe_for_modeling, features)
                    mae, regressor = builder.process_modeling()
                    utility.save_model(athletes_name, activity, best_model_type, best_regressor)
                    if mae < min_mae:
                        min_mae, best_model_type, best_regressor = mae, model_type, regressor
                print('Best model with mean absolute error: {}'.format(min_mae))
                if best_regressor is not None:
                    best_model_dict[activity] = best_model_type

            select_best_model()
            utility.SystemReminder().display_activity_modeling_end(activity, True)

        else:
            utility.SystemReminder().display_activity_modeling_end(activity, False)
    utility.update_trainload_model_types(athletes_name, best_model_dict)


def process_performance_modeling(athletes_name):
    pass


if __name__ == '__main__':
    athletes_names = ['eduardo oliveira', 'xu chen', 'carly hart']
    for athletes_name in athletes_names:
        print('\n\n\n{} {} {} '.format('='*25, athletes_name.title(), '='*25))
        process_train_load_modeling(athletes_name)
        process_performance_modeling(athletes_name)

