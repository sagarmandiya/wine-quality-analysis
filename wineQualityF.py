import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class MyLinearRegression:
    def __init__(self, weight=np.zeros(11), bias=0, learning_rate=0.001, iterations=1000):
        self.weight = weight
        self.bias = bias
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.cost_trend = []
        self.cost = 0

    def predict(self, X):
        X = pd.DataFrame(X)
        x = np.array([X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], X.iloc[:, 3],
                      X.iloc[:, 4], X.iloc[:, 5], X.iloc[:, 6], X.iloc[:, 7],
                      X.iloc[:, 8], X.iloc[:, 9], X.iloc[:, 10]])
        predicted_set = []
        for i in range(len(x)):
            predicted_value = (self.weight[0] * x[0][i] + self.weight[1] * x[1][i] + self.weight[2] * x[2][i] +
                               self.weight[3] * x[3][i] + self.weight[4] * x[4][i] + self.weight[5] * x[5][i] +
                               self.weight[6] * x[6][i] + self.weight[7] * x[7][i] + self.weight[8] * x[8][i] +
                               self.weight[9] * x[9][i] + self.weight[10] * x[10][i] + self.bias)
            predicted_set.append(predicted_value)
        return predicted_set

    def cost_function(self, x, y):
        m = len(y)
        total_error = 0.0
        for i in range(m):
            total_error += (y[i] - (self.weight[0] * x[0][i] + self.weight[1] * x[1][i] + self.weight[2] * x[2][i] +
                                    self.weight[3] * x[3][i] + self.weight[4] * x[4][i] + self.weight[5] * x[5][i] +
                                    self.weight[6] * x[6][i] + self.weight[7] * x[7][i] + self.weight[8] * x[8][i] +
                                    self.weight[9] * x[9][i] + self.weight[10] * x[10][i] + self.bias)) ** 2
        return float(total_error) / (2 * m)

    def update_weights(self, x, y):
        D_theta = [0.0] * 11
        D_bias = 0.0
        count = len(y)
        for i in range(count):
            ypred = (self.weight[0] * x[0][i] + self.weight[1] * x[1][i] + self.weight[2] * x[2][i] +
                     self.weight[3] * x[3][i] + self.weight[4] * x[4][i] + self.weight[5] * x[5][i] +
                     self.weight[6] * x[6][i] + self.weight[7] * x[7][i] + self.weight[8] * x[8][i] +
                     self.weight[9] * x[9][i] + self.weight[10] * x[10][i] + self.bias)
            for j in range(11):
                D_theta[j] += -2 * x[j][i] * (y[i] - ypred)
            D_bias += -2 * (y[i] - ypred)
        for i in range(11):
            self.weight[i] -= (D_theta[i] / count) * self.learning_rate
        self.bias -= (D_bias / count) * self.learning_rate

    def train(self, X, y):
        X = pd.DataFrame(X)
        x = np.array([X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], X.iloc[:, 3],
                      X.iloc[:, 4], X.iloc[:, 5], X.iloc[:, 6], X.iloc[:, 7],
                      X.iloc[:, 8], X.iloc[:, 9], X.iloc[:, 10]])
        for i in range(self.iterations):
            self.update_weights(x, y)
            # Calculating cost
            self.cost = self.cost_function(x, y)
            self.cost_trend.append(self.cost)
            print("Iteration: {}\t Weight: {}\t Bias: {}\t Cost: {}".format(i, self.weight, self.bias, self.cost))


data = pd.read_csv('winequality-red.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

# Fitting multivariate Linear Regression to the Training set
regressor = MyLinearRegression()
regressor.train(X_train, y_train)
epochs = np.arange(regressor.iterations)

plt.plot(epochs, regressor.cost_trend)
plt.xlabel('EPOCHS')
plt.ylabel('COST')
plt.show()

# Predicting the Test set results
y_pred = regressor.predict(X_test)
# Weight: [ 0.19155728 -1.21943446 -1.17271776 -0.79113484 -1.23983212  0.09351752
# -0.08501002 -1.0981176  -0.75430405 -1.11779124  0.66156572]	 Bias: 2.594508124444634	 Cost: 0.3158446162388027
