import sklearn.svm as svm
import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
from utils_p7 import email2TokenList, getVocabDict
import codecs
import SVM_Trainer
import Logic_Regression_Trainer
import nn_trainer
import pytorch_trainer
import Poly_trainer

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
    """Linear kernel

    Args:
        X (np.ndarray): X train dataa
        y (np.ndarray): y train data
        C (float): regularization parameter
    """
    svm_lineal: svm.SVC = svm.SVC(kernel='linear',  C=C)
    svm_lineal.fit(X, y.ravel())
    x1: np.ndarray = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2: np.ndarray = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X1, X2 = np.meshgrid(x1, x2)
    yp: np.ndarray = svm_lineal.predict(
        np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    plt.contour(X1, X2, yp, colors='darkgreen', linewidths=1)
    plt.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], c='b', marker='x')
    plt.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], c='y', marker='o')
    plt.xticks(np.arange(0, 5.5, 0.5))
    plt.yticks(np.arange(1.5, 5.5, 0.5))
    plt.savefig(f'{plot_folder}/SVM_lineal_c{C}.png',  dpi=300)


def kerner_gaussiano(X: np.ndarray, y: np.ndarray, C: float, sigma: float) -> None:
    """Gaussian kernel
    Args:
        X (np.ndarray): X train dataa
        y (np.ndarray): y train data
        C (float): regularization parameter
        sigma (float): scale parameter
    """
    svm_gauss: svm.SVC = svm.SVC(kernel='rbf', C=C, gamma=1/(2*sigma**2))
    svm_gauss.fit(X, y.ravel())
    x1: np.ndarray = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2: np.ndarray = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X1, X2 = np.meshgrid(x1, x2)
    yp: np.ndarray = svm_gauss.predict(
        np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    plt.contour(X1, X2, yp, colors='darkgreen', linewidths=1)
    plt.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], c='b', marker='x')
    plt.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], c='y', marker='o')
    plt.xticks(np.arange(0.0, 1.2, 0.2))
    plt.yticks(np.arange(0.4, 1.1, 0.1))
    plt.savefig(f'{plot_folder}/SVM_gauss_c{C}_sigma{sigma}.png', dpi=300)


def seleccion_sigma_C() -> None:
    """Selects the best C and sigma for the gaussian kernel
    """
    X, y, Xval, yval = load_data3('data/ex6data3.mat')
    C_values: list[float] = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_values: list[float] = C_values
    best_score: float = 0
    best_params: tuple[float] = (0, 0)
    for C in C_values:
        for sigma in sigma_values:
            svm_gauss = svm.SVC(kernel='rbf', C=C, gamma=1/(2*sigma**2))
            svm_gauss.fit(X, y.ravel())
            score = svm_gauss.score(Xval, yval)
            if score > best_score:
                best_score = score
                best_params = (C, sigma)
    print(f'Best score: {best_score}')
    print(f'Best params: {best_params}')
    svm_gauss: svm.SVC = svm.SVC(
        kernel='rbf', C=best_params[0], gamma=1/(2*best_params[1]**2))
    svm_gauss.fit(X, y.ravel())
    x1: np.ndarray = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2: np.ndarray = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X1, X2 = np.meshgrid(x1, x2)
    yp: np.ndarray = svm_gauss.predict(
        np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    plt.contour(X1, X2, yp, colors='darkgreen', linewidths=1)
    plt.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], c='b', marker='x')
    plt.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], c='y', marker='o')
    plt.yticks(np.arange(-0.8, 0.7, 0.2))
    plt.xticks(np.arange(-0.6, 0.4, 0.1))
    plt.savefig(f'{plot_folder}/SVM_gauss_best.png', dpi=300)


def apartado_A() -> None:
    """Apartado A
    """
    X, y = load_data('data/ex6data1.mat')
    print("Linear kernel with C=1")
    kernel_linear(X, y, 1.0)
    plt.clf()
    print("Linear kernel with C=100")
    kernel_linear(X, y, 100.0)
    X, y = load_data('data/ex6data2.mat')
    plt.clf()
    print("Gaussian kernel with C=1 and sigma=0.1")
    kerner_gaussiano(X, y, 1.0, 0.1)
    plt.clf()
    print("Selecting C and sigma for gaussian kernel")
    seleccion_sigma_C()


def load_data_spam() -> list[tuple[list[str], int]]:
    """Loads the spam data
    """
    modes:  list[str] = ['spam', 'easy_ham', 'hard_ham']
    cantidades: list[int] = [500, 2551, 250]
    spam_flag = [1, 0, 0]
    correos = []
    for mode, number, spam in zip(modes, cantidades, spam_flag):
        progress = 0
        length = 50
        for file in range(1, number + 1):
            file = str(file)
            with codecs.open(f'./data_spam/spam/{mode}/{file.zfill(4)}.txt', 'r', encoding='utf-8', errors='ignore') as f:
                progress += 1
                bar_length = int(length * progress / number)
                bar = '[' + '=' * bar_length + \
                    ' ' * (length - bar_length) + ']'
                print(f'\rLoading {mode} {bar} {progress}/{number}', end='')
                email = f.read()
                token_list = email2TokenList(email)
                correos.append((token_list, spam))
        print()
    print(len(correos))
    return correos


def transform_mail(correos, vocab) -> tuple[np.ndarray, np.ndarray]:
    """Transforms the emails into a matrix of length of the vocabulary with 1 if the word is in the email

    Args:
        correos (_type_): mails
        vocab (_type_): dictionary

    Returns:
        tuple[np.ndarray, np.ndarray]: transformed emails with label indicating if its spam or not
    """
    X = []
    y = []

    for c, s in correos:
        x = np.zeros(len(vocab) + 1)
        for word in c:
            if word in vocab:
                x[vocab[word]] = 1
        X.append(x)
        y.append(s)

    return np.array(X), np.array(y)


def plot_results(train_scores: list[float], cv_scores: list[float], test_scores: list[float], times: list[float]) -> None:
    """Plots the results
    Args:
        train_scores (list[float]): train scores
        cv_scores (list[float]): cv scores
        test_scores (list[float]): test scores
        times (list[float]): times
    """
    plt.clf()
    X: np.ndaarray = np.array(
        ['Logistic Regression', 'SVM', 'NN', 'Pytorch', 'Poly'])
    x = np.arange(len(X))
    plt.bar(x-0.2,
            train_scores, 0.2, label=f'Train')
    plt.bar(x,
            cv_scores, 0.2, label=f'CV')
    plt.bar(x+0.2,
            test_scores, 0.2, label=f'Test')
    plt.legend()
    plt.xticks(x, X)
    plt.savefig(f'{plot_folder}/results.png', dpi=300)
    plt.clf()
    plt.plot(X, times)
    plt.savefig(f'{plot_folder}/times.png', dpi=300)


def compare_results() -> None:
    """Compares the results of the different models
    """
    lr_data = sio.loadmat('res/logistic_regression.mat')
    svm_data = sio.loadmat('res/svm.mat')
    nn_data = sio.loadmat('res/nn.mat')
    pytorch_data = sio.loadmat('res/pytorch.mat')
    poly_data = sio.loadmat('res/poly.mat')
    print('Logistic Regression')
    print(f"Score: {lr_data['train_score']}")
    print(f"CV Score: {lr_data['cv_score']}")
    print(f"Test Score: {lr_data['test_score']}")
    print(f"Time: {lr_data['time']}")
    print(f'Best params: {lr_data["best_params"]}')
    print('SVM')
    print(f"Score: {svm_data['train_score']}")
    print(f"CV Score: {svm_data['cv_score']}")
    print(f"Test Score: {svm_data['test_score']}")
    print(f"Time: {svm_data['time']}")
    print(f'Best params: {svm_data["best_params"]}')
    print('NN')
    print(f"Score: {nn_data['train_score']}")
    print(f"CV Score: {nn_data['cv_score']}")
    print(f"Test Score: {nn_data['test_score']}")
    print(f"Time: {nn_data['time']}")
    print(f'Best params: {nn_data["best_params"]}')
    print('Pytorch')
    print(f"Score: {pytorch_data['train_score']}")
    print(f"CV Score: {pytorch_data['cv_score']}")
    print(f"Test Score: {pytorch_data['test_score']}")
    print(f"Time: {pytorch_data['time']}")
    print(f'Best params: {pytorch_data["best_params"]}')
    print('Poly')
    print(f"Score: {poly_data['train_score']}")
    print(f"CV Score: {poly_data['cv_score']}")
    print(f"Test Score: {poly_data['test_score']}")
    print(f"Time: {poly_data['time']}")
    print(f'Best params: {poly_data["best_params"]}')

    train_scores = [lr_data['train_score'][0][0], svm_data['train_score']
                    [0][0], nn_data['train_score'][0][0], pytorch_data['train_score'][0][0], poly_data['train_score'][0][0]]
    cv_scores = [lr_data['cv_score'][0][0], svm_data['cv_score']
                 [0][0], nn_data['cv_score'][0][0], pytorch_data['cv_score'][0][0], poly_data['cv_score'][0][0]]
    test_scores = [lr_data['test_score'][0][0], svm_data['test_score']
                   [0][0], nn_data['test_score'][0][0], pytorch_data['test_score'][0][0], poly_data['test_score'][0][0]]
    times = [lr_data['time'][0][0], svm_data['time'][0][0],
             nn_data['time'][0][0], pytorch_data['time'][0][0], poly_data['time'][0][0]]
    print(train_scores)
    plot_results(train_scores, cv_scores, test_scores, times)


def apartado_B():
    """Apartado B
    """
    correos = load_data_spam()
    vocab = getVocabDict()
    X, y = transform_mail(correos, vocab)
    if not os.path.exists(f'res/svm.mat'):
        print('Training SVM')
        SVM_Trainer.trainer(X, y)
    if not os.path.exists(f'res/logistic_regression.mat'):
        print('Training Logistic Regression')
        Logic_Regression_Trainer.LR_trainer(X, y)
    if not os.path.exists(f'res/pytorch.mat'):
        print('Training Pytorch')
        pytorch_trainer.trainer(X, y)
    if not os.path.exists(f'res/nn.mat'):
        print('Training NN')
        nn_trainer.trainer(X, y)
    if not os.path.exists(f'res/poly.mat'):
        print('Training Poly')
        Poly_trainer.trainer(X, y)

    compare_results()


def main() -> None:
    apartado_A()
    apartado_B()


if __name__ == '__main__':
    main()
