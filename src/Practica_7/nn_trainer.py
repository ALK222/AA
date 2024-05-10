import numpy as np
import scipy.io as sio
import time
from sklearn.model_selection import train_test_split


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """

    g = 1/(1+np.exp(-z))

    return g


def fix_data(X: np.ndarray) -> np.ndarray:
    """Fixes the data to avoid log(0) errors

    Args:
        X (np.ndarray): train data

    Returns:
        np.ndarray: matrix with no 0 or 1 values
    """
    return X + 1e-7


def cost(theta1: np.ndarray, theta2: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float = 0.0) -> float:
    """
    Compute cost for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    """
    L = 2
    layers = [theta1, theta2]
    k: int = y.shape[1]
    h, z = neural_network(X, [theta1, theta2])

    h = h[-1]

    h = fix_data(h)

    J = y * np.log(h + 1e-7)
    J += (1 - y) * np.log(1 - h + 1e-7)

    J = -1 / X.shape[0] * np.sum(J)

    if lambda_ != 0:
        reg = 0
        for layer in layers:
            reg += np.sum(layer[:, 1:] ** 2)
        J += lambda_ / (2 * X.shape[0]) * reg
    return J


def neural_network(X: np.ndarray, thetas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate the neural network with a given set of weights

    Args:
        X (np.ndarray): data
        thetas (np.ndarray): array containing the weights for each layer

    Returns:
        tuple[np.ndarray, np.ndarray]: tuple containing the activations and the z values for each layer
    """
    a = []
    z = []
    a.append(X.copy())
    for theta in thetas:
        a[-1] = np.hstack((np.ones((a[-1].shape[0], 1)), a[-1]))
        z.append(np.dot(a[-1], theta.T))
        a.append(sigmoid(z[-1]))
    return a, z


def backprop(theta1: np.ndarray, theta2: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute cost and gradient for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    grad1 : array_like
        Gradient of the cost function with respect to weights
        for the first layer in the neural network, theta1.
        It has shape (2nd hidden layer size x input size + 1)

    grad2 : array_like
        Gradient of the cost function with respect to weights
        for the second layer in the neural network, theta2.
        It has shape (output layer size x 2nd hidden layer size + 1)

    """
    m = X.shape[0]
    L = 2

    delta = np.empty(2, dtype=object)
    delta[0] = np.zeros(theta1.shape)
    delta[1] = np.zeros(theta2.shape)

    a, z = neural_network(X, [theta1, theta2])

    for k in range(m):
        a1k = a[0][k, :]
        a2k = a[1][k, :]
        hk = a[2][k, :]
        yk = y[k, :]

        d3k = hk - yk
        d2k = np.dot(theta2.T, d3k) * a2k * (1 - a2k)

        delta[0] = delta[0] + \
            np.matmul(d2k[1:, np.newaxis], a1k[np.newaxis, :])
        delta[1] = delta[1] + np.matmul(d3k[:, np.newaxis], a2k[np.newaxis, :])

    grad1 = delta[0] / m
    grad2 = delta[1] / m

    if lambda_ != 0:
        grad1[:, 1:] += lambda_ / m * theta1[:, 1:]
        grad2[:, 1:] += lambda_ / m * theta2[:, 1:]

    J = cost(theta1, theta2, X, y, lambda_)

    return (J, grad1, grad2)


def gradient_descent(X: np.ndarray, y: np.ndarray, theta1: np.ndarray, theta2: np.ndarray, alpha: float, lambda_: float, num_iters: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates the gradient descent for the neural network

    Args:
        X (np.ndarray): Train data
        y (np.ndarray): Expected output in one hot encoding
        theta1 (np.ndarray): initial weights for the first layer
        theta2 (np.ndarray): initial weights for the second layer
        alpha (float): learning rate
        lambda_ (float): regularization parameter
        num_iters (int): number of iterations to run

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: tuple with the final weights for the first and second layer and the cost history
    """
    m = X.shape[0]
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        print('Iteration: ', i + 1, '/', num_iters, end='\r')
        J, grad1, grad2 = backprop(theta1, theta2, X, y, lambda_)
        theta1 = theta1 - alpha * grad1
        theta2 = theta2 - alpha * grad2
        J_history[i] = J
    print('Gradient descent finished.')
    return theta1, theta2, J_history


def train_model(X, y, x_cv, y_cv, alpha, lambda_, num_iters):
    # start = time.time()
    print(f'Alpha: {alpha} Lambda: {lambda_}')
    input_layer_size = X.shape[1]
    hidden_layer_size = 125
    num_labels = 2
    yA = [0 if i == 1 else 1 for i in y]
    yB = [1 if i == 1 else 0 for i in y]
    y_encoded = np.array([yA, yB]).T

    theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1)
    theta2 = np.random.rand(num_labels, hidden_layer_size + 1)

    theta1, theta2, J_history = gradient_descent(
        X, y_encoded, theta1, theta2, alpha, lambda_, num_iters)

    score = predict_percentage(x_cv, y_cv, theta1, theta2)
    # time = time.time() - start
    return (alpha, lambda_, score, theta1, theta2)


def prediction(X: np.ndarray, theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
    """Generates the neural network prediction

    Args:
        X (np.ndarray): data
        theta1 (np.ndarray): first layer weight
        theta2 (np.ndarray): second layer weight

    Returns:
        np.ndarray: best prediction for each row in `X`
    """
    m = X.shape[0]
    p = np.zeros(m)
    a, z = neural_network(X, [theta1, theta2])
    h = a[-1]

    return np.argmax(h, axis=1)


def predict_percentage(X: np.ndarray, y: np.ndarray, theta1: np.ndarray, theta2: np.ndarray) -> float:
    """Gives the accuracy of the neural network

    Args:
        X (ndarray): Train data
        y (ndarray): Expected output
        theta1 (ndarray): First layer weights
        theta2 (ndarray): Second layer weights

    Returns:
        float: Accuracy of the neural network
    """
    m = X.shape[0]
    p = prediction(X, theta1, theta2)

    return p[p == y].size / m


def trainer(X: np.ndarray, y: np.ndarray) -> None:
    lambdas = [0.01, 0.03,  0.1, 0.3, 1, 3, 10, 30]
    alphas = lambdas
    num_iters = 100
    best_score = 0
    best_params = (0, 0)
    input_layer_size = X.shape[1]
    hidden_layer_size = 125
    num_labels = 2
    best_time = 0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3,  random_state=22)
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_test, y_test, test_size=0.5,  random_state=22)
    model = (np.array([]), np.array([]))

    for alpha in alphas:
        for lambda_ in lambdas:

            start = time.time()
            print(f'Alpha: {alpha} Lambda: {lambda_}')
            input_layer_size = X.shape[1]
            hidden_layer_size = 125
            num_labels = 2
            yA = [0 if i == 1 else 1 for i in y]
            yB = [1 if i == 1 else 0 for i in y]
            y_encoded = np.array([yA, yB]).T
            theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1)
            theta2 = np.random.rand(num_labels, hidden_layer_size + 1)

            theta1, theta2, J_history = gradient_descent(
                X, y_encoded, theta1, theta2, alpha, lambda_, num_iters)

            score = predict_percentage(X_cv, y_cv, theta1, theta2)
            print(f'Score: {score}')
            aux_time = time.time() - start
            if score > best_score:
                best_score = score
                best_params = (alpha, lambda_)
                model = (theta1, theta2)
                best_time = aux_time
    print(f'Best score: {best_score}')
    print(f'Best params: {best_params}')

    theta1 = np.random.rand(hidden_layer_size, input_layer_size + 1)
    theta2 = np.random.rand(num_labels, hidden_layer_size + 1)
    yA = [0 if i == 1 else 1 for i in y_train]
    yB = [1 if i == 1 else 0 for i in y_train]
    y_encoded = np.array([yA, yB]).T

    theta1, theta2, = model
    print(f'Training time: {best_time}')

    train_score = predict_percentage(X_train, y_train, theta1, theta2)
    print(f'Train score: {train_score}')
    cv_score = predict_percentage(X_cv, y_cv, theta1, theta2)
    print(f'CV score: {cv_score}')
    test_score = predict_percentage(X_test, y_test, theta1, theta2)
    print(f'Test score: {test_score}')
    sio.savemat('res/nn.mat',
                {'theta1': theta1, 'theta2': theta2, 'train_score': train_score, 'cv_score': cv_score, 'test_score': test_score, 'best_params': best_params, 'time': best_time})
