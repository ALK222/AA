import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from sklearn import svm
import scipy.io as sio
import concurrent.futures


def train_model(C: float, sigma: float, x_train: np.ndarray, y_train: np.ndarray, x_cv: np.ndarray, y_cv: np.ndarray) -> tuple[float, float, float]:
    """Train the model with the given parameters
    Args:
        C (float): Regularization parameter
        sigma (float): Gaussian kernel parameter
        x_train (np.ndarray): Training data
        y_train (np.ndarray): Training target
        x_cv (np.ndarray): Cross validation data
        y_cv (np.ndarray): Cross validation target
    Returns:
        tuple[float, float, float]: Regularization parameter, Gaussian kernel parameter, Score
    """
    print(f'C: {C} sigma: {sigma}')
    svm_gauss = svm.SVC(kernel='rbf', C=C, gamma=1/(2*sigma**2))
    svm_gauss.fit(x_train, y_train.ravel())
    score = svm_gauss.score(x_cv, y_cv.ravel())
    return (C, sigma, score)


def trainer(X: np.ndarray, y: np.ndarray) -> None:
    """Trains the model with the given data
    Args:
        X (np.ndarray): Input data
        y (np.ndarray): Target data
    """
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigma_values = C_values
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3,  random_state=22)
    x_cv, x_test, y_cv, y_test = train_test_split(
        x_test, y_test, test_size=0.5, random_state=22)
    best_score = 0
    best_params = (0, 0)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for C in C_values:
            for sigma in sigma_values:
                futures.append(executor.submit(
                    train_model, C, sigma, x_train, y_train, x_cv, y_cv))

        for future in concurrent.futures.as_completed(futures):
            C, sigma, score = future.result()
            print(f'C: {C} sigma: {sigma} score: {score}')
            if score > best_score:
                best_score = score
                best_params = (C, sigma)

    print(f'Best score: {best_score}')

    start = time.time()
    svm_gauss = svm.SVC(
        kernel='rbf', C=best_params[0], gamma=1/(2*best_params[1]**2))
    svm_gauss.fit(x_train, y_train.ravel())
    end = time.time()
    print(f'Training time: {end-start}')

    test_score = svm_gauss.score(x_test, y_test)
    cv_score = svm_gauss.score(x_cv, y_cv)
    train_score = svm_gauss.score(x_train, y_train)
    sio.savemat('res/svm.mat', {'train_score': train_score,
                                'cv_score': cv_score, 'test_score': test_score, 'best_params': best_params, 'time': end-start})
