import nn
import numpy as np
import scipy.io as sio
import time
import concurrent.futures
from sklearn.model_selection import train_test_split


def train_model(X, y, x_cv, y_cv, alpha, lambda_, num_iters):

    print(f'Alpha: {alpha} Lambda: {lambda_}')
    input_layer_size = X.shape[1]
    hidden_layer_size = 125
    num_labels = 2
    yA = [0 if i == 1 else 1 for i in y]
    yB = [1 if i == 1 else 0 for i in y]
    y_encoded = np.array([yA, yB]).T

    theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1)
    theta2 = np.random.rand(num_labels, hidden_layer_size + 1)

    theta1, theta2, J_history = nn.gradient_descent(
        X, y_encoded, theta1, theta2, alpha, lambda_, num_iters)

    score = nn.predict_percentage(x_cv, y_cv, theta1, theta2)
    return (alpha, lambda_, score)


def trainer(X: np.ndarray, y: np.ndarray):
    lambdas = [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    alphas = [0.1, 0.3, 1, 3, 10, 30]
    num_iters = 1000
    best_score = 0
    best_params = (0, 0)
    input_layer_size = X.shape[1]
    hidden_layer_size = 125
    num_labels = 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3,  random_state=22)
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_test, y_test, test_size=0.5,  random_state=22)
    yA = [0 if i == 1 else 1 for i in y_train]
    yB = [1 if i == 1 else 0 for i in y_train]
    y_encoded = np.array([yA, yB]).T

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = {executor.submit(train_model, X_train, y_train, X_cv, y_cv, alpha, lambda_, num_iters): (
            alpha, lambda_) for alpha in alphas for lambda_ in lambdas}

        for future in concurrent.futures.as_completed(results):
            alpha, lambda_, score = future.result()
            print(f'Alpha: {alpha} Lambda: {lambda_} Score: {score}')
            if score > best_score:
                best_score = score
                best_params = (alpha, lambda_)

    print(f'Best score: {best_score}')
    print(f'Best params: {best_params}')

    start = time.time()
    theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1)
    theta2 = np.random.rand(num_labels, hidden_layer_size + 1)
    theta1, theta2, J_history = nn.gradient_descent(
        X, y_encoded, theta1, theta2, best_params[0], best_params[1], num_iters)
    end = time.time()
    print(f'Training time: {end-start}')

    train_score = nn.predict_percentage(X_train, y_train, theta1, theta2)
    print(f'Train score: {train_score}')
    cv_score = nn.predict_percentage(X_cv, y_cv, theta1, theta2)
    print(f'CV score: {cv_score}')
    test_score = nn.predict_percentage(X_test, y_test, theta1, theta2)
    print(f'Test score: {test_score}')
    sio.savemat('res/nn.mat',
                {'theta1': theta1, 'theta2': theta2, 'train_score': score, 'cv_score': cv_score, 'test_score': test_score, 'best_params': best_params, 'time': end-start})
