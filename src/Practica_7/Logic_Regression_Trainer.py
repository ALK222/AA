import numpy as np
import logistic_reg as lr
import time
import scipy.io as sio
import concurrent.futures
from sklearn.model_selection import train_test_split


def train_model(X, y, x_cv, y_cv, alpha, lambda_, num_iters):
    print(f'Alpha: {alpha} Lambda: {lambda_}')
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    x_cv = np.hstack((np.ones((x_cv.shape[0], 1)), x_cv))
    w = np.zeros(X.shape[1])
    b = 1
    w, b, _, _ = lr.gradient_descent(
        X, y, w, b, lr.compute_cost_reg, lr.compute_gradient_reg, alpha, num_iters, lambda_)
    score = lr.predict_check(x_cv, y_cv, w, b)
    return (alpha, lambda_, score)


def LR_trainer(X: np.ndarray, y: np.ndarray):
    alphas = [0.1, 0.3, 1, 3, 10, 30]
    lambdas = [0.1, 0.3, 1, 3, 10, 30]
    num_iters = 1000
    best_score = 0
    best_params = (0, 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, shuffle=True, random_state=22)
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_test, y_test, test_size=0.5, shuffle=True, random_state=22)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for lambda_ in lambdas:
            for alpha in alphas:
                futures.append(executor.submit(
                    train_model, X_train, y_train, X_cv, y_cv, alpha, lambda_, num_iters))

        for future in concurrent.futures.as_completed(futures):
            alpha, lambda_, score = future.result()
            print(f'Alpha: {alpha} Lambda: {lambda_} Score: {score}')
            if score > best_score:
                best_score = score
                best_params = (alpha, lambda_)

    print(f'Best score: {best_score}')
    print(f'Best params: {best_params}')

    start = time.time()
    w = np.zeros(X.shape[1] + 1)
    b = 1
    X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    w, b, _, _ = lr.gradient_descent(
        X_train, y_train, w, b, lr.compute_cost_reg, lr.compute_gradient_reg, best_params[0], num_iters, best_params[1])
    end = time.time()
    print(f'Training time: {end-start}')
    train_score = lr.predict_check(X_train, y_train, w, b)
    print(f'Train score: {train_score}')
    X_cv = np.hstack((np.ones((X_cv.shape[0], 1)), X_cv))
    cv_score = lr.predict_check(X_cv, y_cv, w, b)
    print(f'CV score: {cv_score}')
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    test_score = lr.predict_check(X_test, y_test, w, b)
    print(f'Test score: {test_score}')
    sio.savemat('res/logistic_regression.mat', {'w': w, 'b': b, 'train_score': train_score,
                'cv_score': cv_score, 'test_score': test_score, 'best_params': best_params, 'time': end-start})
