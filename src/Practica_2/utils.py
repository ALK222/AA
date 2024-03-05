import numpy as np


def load_data():
    data = np.loadtxt("data/ex1data1.txt", delimiter=',')
    X = data[:, 0]
    y = data[:, 1]
    return X, y


def load_data_multi():
    data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
    X_train = data[:, :4]
    y_train = data[:, 4]
    return X_train, y_train