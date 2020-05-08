from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.api import OLS

class ModelBuilder():

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def split_train_validation(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = 0.25, random_state = 25)
        return X_train, X_test, y_train, y_test

    def process_linear_regression(self, X_train, y_train):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        # print('Linear Regression Coefficients', '')
