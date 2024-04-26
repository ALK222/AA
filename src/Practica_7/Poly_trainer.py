import time
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import scipy.io as sio


def cost(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculates the cost of the model

    Args:
        y (np.ndarray): real values
        y_hat (np.ndarray): predicted values

    Returns:
        float: cost of the model
    """
    return np.mean((y_hat - y)**2) / 2


def train_reg(x_train: np.ndarray, y_train: np.ndarray, grado: int, l: float) -> tuple[sp.PolynomialFeatures, sp.StandardScaler, lm.Ridge, np.ndarray]:
    """ Trains a model given the training data with polynomial features and regularization

    Args:
        x_train (np.ndarray): x values of the training data
        y_train (np.ndarray): y values of the training data
        grado (int): degree of the polynomial
        l (float): lambda value for the regularization

    Returns:
        tuple[sp.PolynomialFeatures, sp.StandardScaler, lm.Ridge, np.ndarray]: _description_
    """
    poly: sp.PolynomialFeatures = sp.PolynomialFeatures(
        degree=grado, include_bias=False)
    x_train = poly.fit_transform(x_train)
    scal: sp.StandardScaler = sp.StandardScaler()
    # x_train = scal.fit_transform(x_train)
    model: lm.Ridge = lm.Ridge(alpha=l)
    model.fit(x_train, y_train)
    return poly, scal, model, x_train


def test(x_test: np.ndarray, y_test: np.ndarray, x_train_aux: np.ndarray, y_train: np.ndarray, poly: sp.PolynomialFeatures, scal: sp.StandardScaler, model: Union[lm.LinearRegression, lm.Ridge]) -> tuple[float, float]:
    """Tests the model with the test data

    Args:
        x_test (np.ndarray): x values of the test data
        y_test (np.ndarray): y values of the test data
        x_train_aux (np.ndarray): x values of the training data
        y_train (np.ndarray): y values of the training data
        poly (sp.PolynomialFeatures): polynomial features
        scal (sp.StandardScaler): standard scaler
        model (Union[lm.LinearRegression, lm.Ridge]): model to test

    Returns:
        tuple[float, float]: test cost, train cost
    """
    x_test = poly.transform(x_test)
    # x_test = scal.transform(x_test)

    y_pred_test: np.ndarray = model.predict(x_test)
    test_cost: float = cost(y_test, y_pred_test)

    y_pred_train: np.ndarray = model.predict(x_train_aux)
    train_cost: float = cost(y_train, y_pred_train)

    return test_cost, train_cost


def trainer(X: np.ndarray, y: np.ndarray) -> None:
    """Trains the model with the given data 
    Args:
        X (np.ndarray): Input data
        y (np.ndarray): Target data
    """
    x_train, x_test, y_train, y_test = ms.train_test_split(
        X, y, test_size=0.3, random_state=22)
    x_cv, x_test, y_cv, y_test = ms.train_test_split(
        x_test, y_test, test_size=0.5, random_state=22)

    lambdas: list[float] = [1e-5, 1e-4, 1e-3,
                            1e-2, 1e-1, 1, 10, 100, 300, 600, 900]

    models: np.ndarray = np.empty((16, len(lambdas)), dtype=object)

    min_cost: float = -1
    elec_lambda: float = 0
    eled_grado: int = 0
    costs = np.empty((16, len(lambdas)))

    for i in range(1, 16):
        for l in lambdas:
            pol, scal, model, x_train_aux = train_reg(x_train, y_train, i, l)
            models[i][lambdas.index(l)] = (pol, scal, model, x_train_aux)
            cv_cost, train_cost = test(
                x_cv, y_cv, x_train_aux, y_train, pol, scal, model)
            # costs[i][lambdas.index(l)] = cv_cost
            if min_cost == -1 or cv_cost < min_cost:
                min_cost = cv_cost
                elec_lambda = l
                eled_grado = i
            print(f"Grado: {i} Lambda: {l}-> Cost: {cv_cost}")
    print(f"Grado seleccionado: {eled_grado}")
    print(f"Lambda seleccionado: {elec_lambda}")

    start = time.time()
    pol, scal, model, x_train_aux = train_reg(
        x_train, y_train, eled_grado, elec_lambda)
    end = time.time()

    print(f"Tiempo de entrenamiento: {end-start}")
    X_train_aux = pol.transform(x_train)
    # X_train_aux = scal.transform(X_train_aux)
    y_pred = model.predict(X_train_aux)
    train_pred = (y_pred == y_train).sum() / len(y_train)
    print(f"Train pred: {train_pred}")
    X_cv_aux = pol.transform(x_cv)
    # X_cv_aux = scal.transform(X_cv_aux)
    y_pred = model.predict(X_cv_aux)
    cv_pred = (y_pred == y_cv).sum() / len(y_cv)
    print(f"CV pred: {cv_pred}")
    X_test_aux = pol.transform(x_test)
    # X_test_aux = scal.transform(X_test_aux)
    y_pred = model.predict(X_test_aux)
    test_pred = (y_pred == y_test).sum() / len(y_test)
    print(f"Test pred: {test_pred}")
    sio.savemat('res/poly.mat', {'train_score': train_pred,
                'cv_score': cv_pred, 'test_score': test_pred, 'best_params': (eled_grado, elec_lambda), 'time': end-start})
