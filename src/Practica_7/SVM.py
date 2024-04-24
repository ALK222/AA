import sklearn.svm as svm
import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
from utils_p7 import email2TokenList
import codecs

plot_folder: str = 'memoria/images'


def load_data(file: str) -> tuple[np.ndarray, np.ndarray]:
    """Loads the data from a .mat file

    Args:
        file (str): name of the file

    Returns:
        tuple[np.ndarray, np.ndarray]: X and y data
    """
    data = sio.loadmat(file)
    X = data['X']
    y = data['y']
    return X, y


def load_data3(file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads the data from a .mat file

    Args:
        file (str): name of the file

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X, y, Xval, yval data
    """
    data = sio.loadmat(file)
    X = data['X']
    y = data['y']
    Xval = data['Xval']
    yval = data['yval']
    return X, y, Xval, yval


def kernel_linear(X: np.ndarray, y: np.ndarray, C: float) -> None:
    svm_lineal = svm.SVC(kernel='linear',  C=C)
    svm_lineal.fit(X, y)
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X1, X2 = np.meshgrid(x1, x2)
    yp = svm_lineal.predict(
        np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    plt.contour(X1, X2, yp, colors='darkgreen', linewidths=1)
    plt.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], c='b', marker='x')
    plt.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], c='y', marker='o')
    plt.xticks(np.arange(0, 5.5, 0.5))
    plt.yticks(np.arange(1.5, 5.5, 0.5))
    plt.savefig(f'{plot_folder}/SVM_lineal_c{C}.png',  dpi=300)


def kerner_gaussiano(X: np.ndarray, y: np.ndarray, C: float, sigma: float) -> None:
    svm_gauss = svm.SVC(kernel='rbf', C=C, gamma=1/(2*sigma**2))
    svm_gauss.fit(X, y)
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X1, X2 = np.meshgrid(x1, x2)
    yp = svm_gauss.predict(
        np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    plt.contour(X1, X2, yp, colors='darkgreen', linewidths=1)
    plt.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], c='b', marker='x')
    plt.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], c='y', marker='o')
    plt.xticks(np.arange(0.0, 1.2, 0.2))
    plt.yticks(np.arange(0.4, 1.1, 0.1))
    plt.savefig(f'{plot_folder}/SVM_gauss_c{C}_sigma{sigma}.png', dpi=300)


def seleccion_sigma_C() -> None:
    X, y, Xval, yval = load_data3('data/ex6data3.mat')
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_values = C_values
    best_score = 0
    best_params = (0, 0)
    for C in C_values:
        for sigma in sigma_values:
            svm_gauss = svm.SVC(kernel='rbf', C=C, gamma=1/(2*sigma**2))
            svm_gauss.fit(X, y)
            score = svm_gauss.score(Xval, yval)
            if score > best_score:
                best_score = score
                best_params = (C, sigma)
    print(f'Best score: {best_score}')
    print(f'Best params: {best_params}')
    svm_gauss = svm.SVC(
        kernel='rbf', C=best_params[0], gamma=1/(2*best_params[1]**2))
    svm_gauss.fit(X, y)
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X1, X2 = np.meshgrid(x1, x2)
    yp = svm_gauss.predict(
        np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    plt.contour(X1, X2, yp, colors='darkgreen', linewidths=1)
    plt.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], c='b', marker='x')
    plt.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], c='y', marker='o')
    plt.yticks(np.arange(-0.8, 0.7, 0.2))
    plt.xticks(np.arange(-0.6, 0.4, 0.1))
    plt.savefig(f'{plot_folder}/SVM_gauss_best.png', dpi=300)


def apartado_A():
    X, y = load_data('data/ex6data1.mat')
    kernel_linear(X, y, 1.0)
    plt.clf()
    kernel_linear(X, y, 100.0)
    X, y = load_data('data/ex6data2.mat')
    plt.clf()
    kerner_gaussiano(X, y, 1.0, 0.1)
    plt.clf()
    seleccion_sigma_C()


def apartado_B():
    modes:  list[str] = ['spam', 'easy_ham', 'hard_ham']
    cantidades: list[int] = [500, 2551, 250]
    correos = []

    for mode, number in zip(modes, cantidades):
        for file in range(1, number + 1):
            file = str(file)
            with codecs.open(f'./data_spam/spam/{mode}/{file.zfill(4)}.txt', 'r', encoding='utf-8', errors='ignore') as f:
                print(f'./data_spam/spam/{mode}/{file.zfill(4)}.txt')
                email = f.read()
                token_list = email2TokenList(email)
                correos.append(token_list)
    print(len(correos))


def main():
    apartado_A()
    apartado_B()


if __name__ == '__main__':
    main()
