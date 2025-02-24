import numpy as np

class LinearRegression:
    def __init__(self):
        self.slope = None
        self.bias = None


    def fit(self,x,y):
        """
        learning function
        :param x: independent variable
        :param y: dependent variable
        :return:
        """
        X_mean = np.mean(x)
        y_mean = np.mean(y)

        denominator = np.sum(pow(x-X_mean, 2))
        numerator = np.sum((x-X_mean)*(y-y_mean))

        self.slope = numerator / denominator
        self.bias = y_mean - (self.slope*X_mean)

    def predict(self,x) :
        """
        predict value for input (x)
        :param x:
        :return: list
        """
        return self.slope * np.array(x) + self.bias