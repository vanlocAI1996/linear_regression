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

X, Y = read_data(root, 'data.csv')
x_mean = np.mean(X)
y_mean = np.mean(Y)
#total number of values
n = len(X)
numerator = 0
denominator = 0
for i in range(n):
    numerator += (X[i] - x_mean) * (Y[i] - y_mean)
    denominator += (X[i] - x_mean) ** 2
    
b1 = numerator / denominator
b0 = y_mean - (b1 * x_mean)

print(b1)
print(b0)