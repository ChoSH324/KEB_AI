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


class KNeighborsRegressor:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        저장된 데이터로 모델을 학습
        :param X: 독립 변수 (특징)
        :param y: 종속 변수 (타겟 값)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """
        새로운 입력 데이터에 대한 예측 수행
        :param X: 예측할 입력 데이터
        :return: 예측된 값
        """
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_values = self.y_train[nearest_indices]
            predictions=np.mean(nearest_values)
        return np.array(predictions)