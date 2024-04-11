from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as sp
import sklearn.model_selection as ms
import commandline  # Custom command line parser
import os
import sys

# Constants
# Path to save the plots
plot_folder = "./memoria/plots"
# Random state for reproducibility
RANDOM_STATE = 1


def plot_dataset(x: np.ndarray, y: np.ndarray, x_ideal: np.ndarray, y_ideal: np.ndarray, name: str) -> None:
    """Plots the dataset and the ideal data

    Args:
        x (np.ndarray): x values of the dataset with noise
        y (np.ndarray): y values of the dataset with noise
        x_ideal (np.ndarray): x ideal values of the dataset
        y_ideal (np.ndarray): y ideal values of the dataset
        name (str): name of the file
    """
    plt.plot(x, y, 'o', label='Train data')
    plt.plot(x_ideal, y_ideal, label='Ideal data', c='red')
    plt.legend()
    plt.savefig(f'{plot_folder}/{name}.png', dpi=300)
    plt.clf()


def plot_linear_data(x: np.ndarray, y: np.ndarray, x_ideal: np.ndarray, y_ideal: np.ndarray, model_range: np.ndarray, model: np.ndarray, name: str) -> None:
    """Plots the dataset, the ideal data and the model

    Args:
        x (np.ndarray): x values of the dataset with noise
        y (np.ndarray): y values of the dataset with noise
        x_ideal (np.ndarray): x ideal values of the dataset
        y_ideal (np.ndarray): y ideal values of the dataset
        model_range (np.ndarray): x values of the model
        model (np.ndarray): y values of the model
        name (str): file name
    """
    plt.plot(x, y, 'o', label='Train data', markersize=2)
    plt.plot(x_ideal, y_ideal, label='Ideal data',
             c='red', linestyle='dashed', linewidth=1.5)
    plt.plot(model_range, model, label='Predictions', c='blue', linewidth=1)
    plt.legend()
    plt.savefig(f'{plot_folder}/{name}.png', dpi=300)
    plt.clf()


def draw_learning_curve(x: np.ndarray, y: np.ndarray, x_v: np.ndarray, y_v: np.ndarray):
    """Plots the learning curve of the model based on the number of training sample sizes

    Args:
        x (np.ndarray): number of samples
        y (np.ndarray): cost of the training set
        x_v (np.ndarray): number of samples
        y_v (np.ndarray): cost of the validation set
    """
    plt.figure()
    plt.plot(x_v, y_v, c='orange', label='validation')
    plt.plot(x, y, c='blue', label='train')
    plt.legend()
    plt.savefig(f'{plot_folder}/learning_curve.png', dpi=300)


def gen_data(m: int, seed: int = 1, scale: float = 0.7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates a dataset with noise

    Args:
        m (int): number of samples
        seed (int, optional): random seed. Defaults to 1.
        scale (float, optional): scale of the noise. Defaults to 0.7.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: x_train, y_train, x_ideal, y_ideal
    """
    c: int = 0
    x_train: np.ndarray = np.linspace(0, 49, m)
    np.random.seed(seed)
    y_ideal: np.ndarray = x_train**2 + c
    y_train: np.ndarray = y_ideal + scale * \
        y_ideal * (np.random.sample((m,)) - 0.5)
    x_ideal: np.ndarray = x_train
    return x_train, y_train, x_ideal, y_ideal


def cost(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculates the cost of the model

    Args:
        y (np.ndarray): real values
        y_hat (np.ndarray): predicted values

    Returns:
        float: cost of the model
    """
    return np.mean((y_hat - y)**2) / 2


def train(x_train: np.ndarray, y_train: np.ndarray, grado: int) -> tuple[sp.PolynomialFeatures, sp.StandardScaler, lm.LinearRegression, np.ndarray]:
    """ Trains a model given the training data with polynomial features

    Args:
        x_train (np.ndarray): x values of the training data
        y_train (np.ndarray): y values of the training data
        grado (int): degree of the polynomial

    Returns:
        tuple[sp.PolynomialFeatures, sp.StandardScaler, lm.LinearRegression, np.ndarray]: _description_
    """
    poly: sp.PolynomialFeatures = sp.PolynomialFeatures(
        degree=grado, include_bias=False)
    x_train = poly.fit_transform(x_train[:, None])
    scal: sp.StandardScaler = sp.StandardScaler()
    x_train = scal.fit_transform(x_train)
    model: lm.LinearRegression = lm.LinearRegression()
    model.fit(x_train, y_train)
    return poly, scal, model, x_train


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
    x_train = poly.fit_transform(x_train[:, None])
    scal: sp.StandardScaler = sp.StandardScaler()
    x_train = scal.fit_transform(x_train)
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
    x_test = poly.transform(x_test[:, None])
    x_test = scal.transform(x_test)

    y_pred_test: np.ndarray = model.predict(x_test)
    test_cost: float = cost(y_test, y_pred_test)

    y_pred_train: np.ndarray = model.predict(x_train_aux)
    train_cost: float = cost(y_train, y_pred_train)

    return test_cost, train_cost


def overfitting(x: np.ndarray, y: np.ndarray, x_i: np.ndarray, y_i: np.ndarray) -> None:
    """Tests the overfitting of the model

    Args:
        x (np.ndarray): x values of the dataset with noise
        y (np.ndarray): y values of the dataset with noise
        x_i (np.ndarray): x ideal values of the dataset
        y_i (np.ndarray): y ideal values of the dataset
    """
    print("Overfitting")
    x_train, x_test, y_train, y_test = ms.train_test_split(
        x, y, test_size=0.33, random_state=RANDOM_STATE)
    pol, scal, model, x_train_aux = train(x_train, y_train, 15)
    range_x: np.ndarray = np.linspace(np.min(x), np.max(x), 1000)
    range_x = range_x[:, None]
    range_x_p: np.ndarray = pol.transform(range_x)
    range_x_p = scal.transform(range_x_p)
    y_pred: np.ndarray = model.predict(range_x_p)
    plot_linear_data(x, y, x_i, y_i, range_x, y_pred, 'overfitting')
    test_cost, train_cost = test(
        x_test, y_test, x_train_aux, y_train, pol, scal, model)

    print(f"Train cost: {train_cost}")
    print(f"Test cost: {test_cost}")


def seleccion_grado(x: np.ndarray, y: np.ndarray, x_i: np.ndarray, y_i: np.ndarray) -> None:
    """Selects the best degree for the model

    Args:
        x (np.ndarray): x values of the dataset with noise
        y (np.ndarray): y values of the dataset with noise
        x_i (np.ndarray): x ideal values of the dataset
        y_i (np.ndarray): y ideal values of the dataset
    """
    print("Seleccion de grado")
    x_train, x_test, y_train, y_test = ms.train_test_split(
        x, y, test_size=0.4, random_state=RANDOM_STATE)
    x_test, x_cv, y_test, y_cv = ms.train_test_split(
        x_test, y_test, test_size=0.5, random_state=RANDOM_STATE)
    min_cost: float = 0
    min_grado: float = 0
    models: np.ndarray = np.empty(10, dtype=object)
    for grado in range(10):
        pol, scal, model, x_train_aux = train(x_train, y_train, grado + 1)
        cv_cost, train_cost = test(
            x_cv, y_cv, x_train_aux, y_train, pol, scal, model)
        models[grado] = (pol, scal, model, x_train_aux)
        if min_cost == 0 or cv_cost < min_cost:
            min_cost = cv_cost
            min_grado = grado + 1
    print(f"Grado seleccionado: {min_grado}")

    x_range: np.ndarray = np.linspace(np.min(x), np.max(x), 1000)
    x_range: np.ndarray = x_range[:, None]
    x_range_p: np.ndarray = models[min_grado - 1][0].transform(x_range)
    x_range_p: np.ndarray = models[min_grado - 1][1].transform(x_range_p)
    y_pred: np.ndarray = models[min_grado - 1][2].predict(x_range_p)

    plot_linear_data(x, y, x_i, y_i, x_range, y_pred, 'grado')

    test_cost, train_cost = test(
        x_test, y_test, models[min_grado - 1][3], y_train, models[min_grado - 1][0], models[min_grado - 1][1], models[min_grado - 1][2])
    print(f"Train cost: {train_cost}")
    print(f"CV cost: {min_cost}")
    print(f"Test cost: {test_cost}")


def seleccion_lambda(x: np.ndarray, y: np.ndarray, x_i: np.ndarray, y_i: np.ndarray) -> None:
    """Selects the best lambda for the model

    Args:
        x (np.ndarray): x values of the dataset with noise
        y (np.ndarray): y values of the dataset with noise
        x_i (np.ndarray): x ideal values of the dataset
        y_i (np.ndarray): y ideal values of the dataset
    """
    print("Seleccion de lambda")
    lambdas: list[float] = [1e-6, 1e-5, 1e-4, 1e-3,
                            1e-2, 1e-1, 1, 10, 100, 300, 600, 900]
    alpha: float = 0
    min_cost: float = -1
    x_train, x_test, y_train, y_test = ms.train_test_split(
        x, y, test_size=0.4, random_state=RANDOM_STATE)
    x_test, x_cv, y_test, y_cv = ms.train_test_split(
        x_test, y_test, test_size=0.5, random_state=RANDOM_STATE)
    models: np.ndarray = np.empty(len(lambdas), dtype=object)

    for l in lambdas:
        pol, scal, model, x_train_aux = train_reg(x_train, y_train, 15, l)
        test_cost, train_cost = test(
            x_cv, y_cv, x_train_aux, y_train, pol, scal, model)
        models[lambdas.index(l)] = (pol, scal, model, x_train_aux)
        if min_cost == -1 or test_cost < min_cost:
            min_cost = test_cost
            alpha = l
        print(f"Lambda: {l}-> Cost: {test_cost}")
    print(f"Lamda seleccionado: {alpha}")

    x_range: np.ndarray = np.linspace(np.min(x), np.max(x), 1000)
    x_range = x_range[:, None]
    x_range_p: np.ndarray = models[lambdas.index(alpha)][0].transform(x_range)
    x_range_p = models[lambdas.index(alpha)][1].transform(x_range_p)
    y_pred: np.ndarray = models[lambdas.index(alpha)][2].predict(x_range_p)

    plot_linear_data(x, y, x_i, y_i, x_range, y_pred, 'lambda')

    test_cost, train_cost = test(
        x_test, y_test, models[lambdas.index(alpha)][3], y_train, models[lambdas.index(alpha)][0], models[lambdas.index(alpha)][1], models[lambdas.index(alpha)][2])
    print(f"Train cost: {train_cost}")
    print(f"CV cost: {min_cost}")
    print(f"Test cost: {test_cost}")


def seleccion_hiperparametros(x: np.ndarray, y: np.ndarray, x_i: np.ndarray, y_i: np.ndarray) -> None:
    """Selects the best hyperparameters for the model

    Args:
        x (np.ndarray): x values of the dataset with noise
        y (np.ndarray): y values of the dataset with noise
        x_i (np.ndarray): x ideal values of the dataset
        y_i (np.ndarray): y ideal values of the dataset
    """
    print("Seleccion de hiperparametros")
    x_train, x_test, y_train, y_test = ms.train_test_split(
        x, y, test_size=0.4, random_state=RANDOM_STATE)
    x_test, x_cv, y_test, y_cv = ms.train_test_split(
        x_test, y_test, test_size=0.5, random_state=RANDOM_STATE)

    lambdas: list[float] = [1e-6, 1e-5, 1e-4, 1e-3,
                            1e-2, 1e-1, 1, 10, 100, 300, 600, 900]

    models: np.ndarray = np.empty((16, len(lambdas)), dtype=object)

    min_cost: float = -1
    elec_lambda: float = 0
    eled_grado: int = 0

    for i in range(1, 15):
        for l in lambdas:
            pol, scal, model, x_train_aux = train_reg(x_train, y_train, i, l)
            models[i][lambdas.index(l)] = (pol, scal, model, x_train_aux)
            cv_cost, train_cost = test(
                x_cv, y_cv, x_train_aux, y_train, pol, scal, model)
            if min_cost == -1 or cv_cost < min_cost:
                min_cost = cv_cost
                elec_lambda = l
                eled_grado = i
            print(f"Grado: {i} Lambda: {l}-> Cost: {cv_cost}")
    print(f"Grado seleccionado: {eled_grado}")
    print(f"Lambda seleccionado: {elec_lambda}")

    x_range: np.ndarray = np.linspace(np.min(x), np.max(x), 10000)
    x_range = x_range[:, None]
    x_range_p: np.ndarray = models[eled_grado][lambdas.index(
        elec_lambda)][0].transform(x_range)
    x_range_p = models[eled_grado][lambdas.index(
        elec_lambda)][1].transform(x_range_p)
    y_pred: np.ndarray = models[eled_grado][lambdas.index(
        elec_lambda)][2].predict(x_range_p)

    plot_linear_data(x, y, x_i, y_i, x_range, y_pred,
                     'hiperparametros')

    test_cost, train_cost = test(
        x_test, y_test, models[eled_grado][lambdas.index(elec_lambda)][3], y_train, models[eled_grado][lambdas.index(elec_lambda)][0], models[eled_grado][lambdas.index(elec_lambda)][1], models[eled_grado][lambdas.index(elec_lambda)][2])
    print(f"Train cost: {train_cost}")
    print(f"CV cost: {min_cost}")
    print(f"Test cost: {test_cost}")


def learning_curve():
    """Plots the learning curve of the model based on the number of training sample sizes
    """
    print("Learning curve")
    J_cost_train: list[float] = []
    J_const_val: list[float] = []
    params: list[int] = range(50, 601, 50)
    x, y, x_i, y_i = gen_data(1000)
    x_train_t, x_test, y_train_t, y_test = ms.train_test_split(
        x, y, test_size=0.4, random_state=RANDOM_STATE)
    x_test, x_cv, y_test, y_cv = ms.train_test_split(
        x_test, y_test, test_size=0.5, random_state=RANDOM_STATE)

    for i in range(len(params)):
        indexes = np.linspace(
            0, len(x_train_t) - 1, params[i], dtype=int)
        x_train = x_train_t[indexes]
        y_train = y_train_t[indexes]
        pol, scal, model, x_train_aux = train(x_train, y_train, 16)
        cv_cost, train_cost = test(
            x_cv, y_cv, x_train_aux, y_train, pol, scal, model)
        J_cost_train.append(train_cost)
        J_const_val.append(cv_cost)

    draw_learning_curve(params, J_cost_train, params, J_const_val)

    print(f'Mejor valor de entrenamiento: {np.min(J_cost_train)}')
    print(f'Mejor valor de validacion: {np.min(J_const_val)}')


def test_overfitting(x: np.ndarray, y: np.ndarray, x_i: np.ndarray, y_i: np.ndarray, commandLine: commandline.CommandLine) -> None:
    """Tests the overfitting of the model

    Args:
        x (np.ndarray): x values of the dataset with noise
        y (np.ndarray): y values of the dataset with noise
        x_i (np.ndarray): x ideal values of the dataset
        y_i (np.ndarray): y ideal values of the dataset
        commandLine (commandline.CommandLine): command line arguments
    """
    if (not os.path.exists(f'{plot_folder}/overfitting.png') or commandLine.overfitting or commandLine.all):
        overfitting(x, y, x_i, y_i)
    else:
        if (commandLine.interactive):
            answer = ''
            while (answer != 'y' and answer != 'n'):
                print("Recreate overfitting test? [y/n]")
                answer = input()
            if (answer == 'y'):
                overfitting(x, y, x_i, y_i)


def test_seleccion_grado(x: np.ndarray, y: np.ndarray, x_i: np.ndarray, y_i: np.ndarray, commandLine: commandline.CommandLine) -> None:
    """Selects the best degree for the model
    """
    if (not os.path.exists(f'{plot_folder}/grado.png') or commandLine.grado or commandLine.all):
        seleccion_grado(x, y, x_i, y_i)
    else:
        if (commandLine.interactive):
            answer = ''
            while (answer != 'y' and answer != 'n'):
                print("Recreate grado test? [y/n]")
                answer = input()
            if (answer == 'y'):
                seleccion_grado(x, y, x_i, y_i)


def test_regularizacion(x: np.ndarray, y: np.ndarray, x_i: np.ndarray, y_i: np.ndarray, commandLine: commandline.CommandLine) -> None:
    """Selects the best lambda for the model
    """
    if (not os.path.exists(f'{plot_folder}/lambda.png') or commandLine.reg or commandLine.all):
        seleccion_lambda(x, y, x_i, y_i)
    else:
        if (commandLine.interactive):
            answer = ''
            while (answer != 'y' and answer != 'n'):
                print("Recreate regularizacion test? [y/n]")
                answer = input()
            if (answer == 'y'):
                seleccion_lambda(x, y, x_i, y_i)


def test_hiperparametros(x: np.ndarray, y: np.ndarray, x_i: np.ndarray, y_i: np.ndarray, commandLine: commandline.CommandLine) -> None:
    """Selects the best hyperparameters for the model
    """
    if (not os.path.exists(f'{plot_folder}/hiperparametros.png') or commandLine.hyperparam or commandLine.all):
        seleccion_hiperparametros(x, y, x_i, y_i)
    else:
        if (commandLine.interactive):
            answer = ''
            while (answer != 'y' and answer != 'n'):
                print("Recreate hiperparametros test? [y/n]")
                answer = input()
            if (answer == 'y'):
                seleccion_hiperparametros(x, y, x_i, y_i)


def test_learning_curve(x: np.ndarray, i: np.ndarray, x_i: np.ndarray, y_i: np.ndarray, commandLine: commandline.CommandLine) -> None:
    """Plots the learning curve of the model based on the number of training sample sizes
    """
    if (not os.path.exists(f'{plot_folder}/learning_curve.png') or commandLine.learning_curve or commandLine.all):
        learning_curve()
    else:
        if (commandLine.interactive):
            answer = ''
            while (answer != 'y' and answer != 'n'):
                print("Recreate learning curve test? [y/n]")
                answer = input()
            if (answer == 'y'):
                learning_curve()


def prepare_folder() -> None:
    """Creates the folder to save the plots
    """
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)


def main() -> None:
    """Main function
    """
    commandLine = commandline.CommandLine()
    commandLine.parse(sys.argv[1:])
    x, y, x_i, y_i = gen_data(64)
    funcs = [test_overfitting, test_seleccion_grado,
             test_regularizacion, test_hiperparametros, test_learning_curve]

    prepare_folder()
    if not os.path.exists(f'{plot_folder}/dataset.png'):
        plot_dataset(x, y, x_i, y_i, 'dataset')

    for func in funcs:
        if func.__name__ == 'test_hiperparametros':
            x, y, x_i, y_i = gen_data(750)
        func(x, y, x_i, y_i, commandLine)


if __name__ == '__main__':
    main()
