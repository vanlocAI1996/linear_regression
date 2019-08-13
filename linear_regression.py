import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

root = 'data'
def read_data(root, filename):
	csv_path = os.path.join(root, filename)
	df = pd.read_csv(csv_path)
	X = df['x']
	y = df['y']
	return X, y

def plot_scatter(X, y):
	plt.scatter(X, y)
	plt.show()

def hypothesis(X, w):
	return np.dot(X, w)

def calculate_cost(y, h):
	m = np.shape(y)[0]
	return np.sum(np.power(y - h, 2)) / m

def fit(X, y, leaning_rate=0.0001, iterations=100):
	m, n = np.shape(X)
	ones = np.ones((m, 1))
	X = np.concatenate((ones, X), axis=1)
	w_init = np.random.rand(1, n + 1).T
	w = [w_init]
	cost_histories = []
	for iteration in range(iterations):
		h = hypothesis(X, w[-1])
		loss = h - y
		gradient = leaning_rate * np.dot(X.T, loss) / m
		w_new = w[-1] - gradient
		cost = calculate_cost(y, h)
		cost_histories.append(cost)
		w.append(w_new)
	return w, cost_histories	

def draw_cost(cost_histories):
	plt.xlabel('x')
	plt.ylabel('y')
	iterations = len(cost_histories)
	plt.plot(range(iterations), cost_histories, 'b.')
	plt.show()

def predict(X_test, w):
	return [w[0][1]*x + w[0][0] for x in X_test]

X, y = read_data('data', 'data.csv')

X = X.values.reshape(-1, 1)
y = y.values.reshape(-1, 1)
w, his = fit(X, y)
print(w[-1])
# X_test = np.linspace(0, 100, 100)
# predict = predict(X_test, w)
# plt.scatter(X, y)
# plt.plot(X_test, predict)
# plt.show()
# print(predict)
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X, y)
print(reg.coef_)
print(reg.intercept_ )