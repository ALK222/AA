import numpy as np
import copy
import matplotlib.pyplot as plt
import public_tests
import utils
import csv


def fun(x: np.ndarray, w: np.ndarray, b: float) -> float:
    """Returns a predicted w value for a linear regression with the given w and b values.

    Args:
        x (np.ndarray): x value for de function
        w (np.ndarray): array of w
        b (float): b value

    Returns:
        float: predicted y value
    """

    return np.dot(x, w) + b


def zscore_normalize_features(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (np.array(X) - mu) / sigma

    return (X_norm, mu, sigma)


def compute_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
    Returns
      cost (scalar)    : cost
    """
    x_fun = fun(X, w, b)

    cost = np.sum((x_fun - y)**2)

    cost = cost / (2 * X.shape[0])

    return cost


def compute_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> tuple[float, np.ndarray]:
    """
    Computes the gradient for linear regression 
    Args:
      X : (ndarray Shape (m,n)) matrix of examples 
      y : (ndarray Shape (m,))  target value of each example
      w : (ndarray Shape (n,))  parameters of the model      
      b : (scalar)              parameter of the model      
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """

    x_fun = fun(X, w, b)
    dj_dw = np.dot((x_fun - y), X) / X.shape[0]

    dj_db: float = np.sum(x_fun - y)

    dj_db /= X.shape[0]

    return dj_db, dj_dw


def gradient_descent(X: np.ndarray, y: np.ndarray, w_in: np.ndarray, b_in: float, cost_function: float,
                     gradient_function: float, alpha: float, num_iters: int) -> tuple[np.ndarray, float, list[float]]:
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X : (array_like Shape (m,n)    matrix of examples 
      y : (array_like Shape (m,))    target value of each example
      w_in : (array_like Shape (n,)) Initial values of parameters of the model
      b_in : (scalar)                Initial value of parameter of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (array_like Shape (n,)) Updated values of parameters of the model
          after running gradient descent
      b : (scalar)                Updated value of parameter of the model 
          after running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """

    w = copy.deepcopy(w_in)
    b = 0 + b_in
    J_history = [cost_function(X, y, w, b)]

    for i in range(0, num_iters):
        (dj_db, dj_dw) = gradient_function(X, y, w, b)

        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
        J_history.append(cost_function(X, y, w, b))

    return w, b, J_history


def visualize_data(X_train: np.ndarray, y_train: np.ndarray, name: str) -> None:
    """Generates a scatter plot of the data

    Args:
        X_train (np.ndarray): training data
        y_train (np.ndarray): training data
        name (str): name of the plot
    """
    X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
    fig, ax = plt.subplots(1, 4, figsize=(25, 5), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:, i], y_train)
        ax[i].set_xlabel(X_features[i])
        ax[0].set_ylabel("Price (1000's)")
    plt.savefig(f'./memoria/images/{name}.png', dpi=300)


def visualize_data_train(X_train: np.ndarray, y_train: np.ndarray, prediction: np.ndarray) -> None:
    """Generates a scatter data with the original data and the predictions

    Args:
        X_train (np.ndarray): Train values
        y_train (np.ndarray): Train prices
        prediction (np.ndarray): Predicted prices
    """
    X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
    fig, ax = plt.subplots(1, 4, figsize=(25, 5), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:, i], y_train)
        ax[i].scatter(X_train[:, i], prediction, color='orange')
        ax[i].set_xlabel(X_features[i])
        ax[0].set_ylabel("Price (1000's)")
    plt.savefig(f'./memoria/images/predicted_data.png', dpi=300)


def visualize_partial_regressions(X_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, b: float) -> None:
    """Draws partial regressions for each variable

    Args:
        X_train (np.ndarray): Train data
        y_train (np.ndarray): Train prices
        w (np.ndarray): Weights
        b (float): bias
    """
    (X_norm, _, _) = zscore_normalize_features(X_train)
    X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
    fig, ax = plt.subplots(1, 4, figsize=(25, 5), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:, i], y_train)
        ax[i].plot(X_train[:, i], fun(
            X_norm[:, i], w[i], b), color='red')
        ax[i].set_xlabel(X_features[i])
        ax[0].set_ylabel("Price (1000's)")
    plt.savefig(f'./memoria/images/partail_regression.png', dpi=300)


def visualize_J_history(J_history: np.ndarray) -> None:
    """Generates a plot with the evolution of the cost function

    Args:
        J_history (np.ndarray): Cost function over each iteration
    """
    plt.clf()
    plt.figure(figsize=(7, 5))
    plt.plot(J_history)
    plt.xlabel('Iterations')
    plt.xscale('log')
    plt.ylabel('Cost')
    plt.title('Cost function over iterations')
    plt.savefig(f'./memoria/images/cost_function.png', dpi=300)


def write_results(J_history: np.ndarray, w: np.ndarray, b: float, mu: np.ndarray, sigma: np.ndarray) -> None:
    X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
    with open('./memoria/recursos/J_history_simplificado.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteracion', 'J'])
        slicedHistory = J_history[0:-1:100]
        for i in range(1, len(slicedHistory) + 1):
            writer.writerow([i*100, slicedHistory[i - 1]])
    with open('./memoria/recursos/wb.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(X_features + ['b']),
        writer.writerow(np.concatenate((w, [b])))

    with open('./memoria/recursos/mean.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(X_features),
        writer.writerow(mu)
    with open('./memoria/recursos/sigma.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(X_features),
        writer.writerow(sigma)


def main():
    public_tests.compute_cost_test(compute_cost)
    public_tests.compute_gradient_test(compute_gradient)
    (X, y) = utils.load_data_multi()

    visualize_data(X, y, 'visualizacion_inicial')

    (X_norm, mu, sigma) = zscore_normalize_features(X)

    (w, b, J_history) = gradient_descent(
        X_norm, y, np.ones(shape=X.shape[1]), 0, compute_cost, compute_gradient, 0.1, 1000)

    # print(w, b, J_history)
    data = np.array([1200, 3, 1, 40])
    data = (data - mu) / sigma
    print(fun(data, w, b) * 1000)

    predicted_y = fun(X_norm, w, b)

    visualize_data_train(X, y, predicted_y)
    visualize_partial_regressions(X, y, w, b)
    visualize_J_history(J_history)
    write_results(J_history, w, b, mu, sigma)


if __name__ == ("__main__"):
    main()
