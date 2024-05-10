import numpy as np
import copy
import time
import scipy.io as sio
import concurrent.futures
from sklearn.model_selection import train_test_split


def compute_cost_reg(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 1) -> float:
    """
    Computes the cost over all examples
    Args:
      X : (array_like Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : (array_like Shape (n,)) Values of bias parameter of the model
      lambda_ : (scalar, float)    Controls amount of regularization
    Returns:
      total_cost: (scalar)         cost 
    """

    total_cost = compute_cost(X, y, w, b)
    total_cost += (lambda_ / (2 * X.shape[0])) * np.sum(w**2)

    return total_cost


def loss(X: np.ndarray, Y: np.ndarray, fun: np.ndarray, w: np.ndarray, b: float) -> float:
    """loss function for the logistic regression

    Args:
        X (np.ndarray): X values
        Y (np.ndarray): Expected y results
        fun (np.ndarray): logistic regression function
        w (np.ndarray): weights
        b (float): bias

    Returns:
        float: total loss of the regression
    """

    return (-Y * np.log(fun(X, w, b) + 1e-6)) - ((1 - Y) * np.log(1 - fun(X, w, b) + 1e-6))


def compute_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_=None) -> float:
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value
      w : (array_like Shape (n,)) Values of parameters of the model
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost
    """
    # apply the loss function for each element of the x and y arrays
    loss_v = loss(X, y, function, w, b)
    total_cost = np.sum(loss_v)
    total_cost /= X.shape[0]

    return total_cost


def compute_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_=None) -> tuple[float, np.ndarray]:
    """
    Computes the gradient for logistic regression

    Args:
        X : (ndarray Shape (m,n)) variable such as house size
        y : (array_like Shape (m,1)) actual value
        w : (array_like Shape (n,1)) values of parameters of the model
        b : (scalar)                 value of parameter of the model
        lambda_: unused placeholder
    Returns
        dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b.
        dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w.
    """

    func = function(X, w, b)

    dj_dw = np.dot(func - y, X)
    dj_dw /= X.shape[0]

    dj_db = np.sum(func - y)
    dj_db /= X.shape[0]

    return dj_db, dj_dw


def compute_gradient_reg(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 1) -> tuple[float, np.ndarray]:
    """
    Computes the gradient for linear regression 

    Args:
      X : (ndarray Shape (m,n))   variable such as house size 
      y : (ndarray Shape (m,))    actual value 
      w : (ndarray Shape (n,))    values of parameters of the model      
      b : (scalar)                value of parameter of the model  
      lambda_ : (scalar,float)    regularization constant
    Returns
      dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """
    dj_db, dj_dw = compute_gradient(X, y, w, b)
    dj_dw += (lambda_ / X.shape[0]) * w

    return dj_db, dj_dw


def gradient_descent(X: np.ndarray, y: np.ndarray, w_in: np.ndarray, b_in: float, cost_function: float, gradient_function: float, alpha: float, num_iters: int, lambda_: float = None) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant

    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """

    w = copy.deepcopy(w_in)
    b = b_in
    predict_history = [predict_check(X, y, w, b)]
    J_history = [cost_function(X, y, w, b, lambda_)]

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b, lambda_)
        w = w - (alpha * dj_dw)
        b -= alpha * dj_db
        J_history.append(cost_function(X, y, w, b, lambda_))
        predict_history.append(predict_check(X, y, w, b))

    return w, b, np.array(J_history), predict_history


def predict(X, w, b) -> np.ndarray:
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w and b

    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5
    """

    p = np.vectorize(lambda x: 1 if x > 0.5 else 0)(
        function(X, w, b))
    return p


def predict_check(X, Z, w, b) -> float:
    """Gives a percentage of the accuracy of the prediction

    Args:
        X (_type_): X train data
        Z (_type_): expected values
        w (_type_): weights
        b (_type_): bias

    Returns:
        float: percentage of accuracy
    """
    p = predict(X, w, b)
    return np.sum(p == Z) / Z.shape[0]


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


def function(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Function using ''sigmoid'' to calculate the value of y to the given x, w and b

    Args:
        x (np.ndarray): X data
        w (np.ndarray): w data
        b (float): b data

    Returns:
        np.ndarray: final value after the sigmoid
    """
    return sigmoid(np.dot(x, w) + b)


def train_model(X: np.ndarray, y: np.ndarray, x_cv: np.ndarray, y_cv: np.ndarray, alpha: float, lambda_: float, num_iters: int) -> tuple[float, float, float]:
    """Train the model with the given parameters
    Args:
        X (np.ndarray): Training data
        y (np.ndarray): Training target
        x_cv (np.ndarray): Cross validation data
        y_cv (np.ndarray): Cross validation target
        alpha (float): Learning rate
        lambda_ (float): Regularization parameter
        num_iters (int): Number of iterations
    Returns:
        tuple[float, float, float]: Learning rate, Regularization parameter, Score
    """
    print(f'Alpha: {alpha} Lambda: {lambda_}')
    m, n = X.shape
    X = np.hstack((np.ones((m, 1)), X))
    x_cv = np.hstack((np.ones((x_cv.shape[0], 1)), x_cv))
    w = np.zeros(X.shape[1])
    b = 1
    w, b, _, _ = gradient_descent(
        X, y, w, b, compute_cost_reg, compute_gradient_reg, alpha, num_iters, lambda_)
    score = predict_check(x_cv, y_cv, w, b)
    return (alpha, lambda_, score)


def LR_trainer(X: np.ndarray, y: np.ndarray) -> None:
    """Trains the model with the given data
    Args:
        X (np.ndarray): Input data
        y (np.ndarray): Target data
    """
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
    w, b, _, _ = gradient_descent(
        X_train, y_train, w, b, compute_cost_reg, compute_gradient_reg, best_params[0], num_iters, best_params[1])
    end = time.time()
    print(f'Training time: {end-start}')
    train_score = predict_check(X_train, y_train, w, b)
    print(f'Train score: {train_score}')
    X_cv = np.hstack((np.ones((X_cv.shape[0], 1)), X_cv))
    cv_score = predict_check(X_cv, y_cv, w, b)
    print(f'CV score: {cv_score}')
    X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
    test_score = predict_check(X_test, y_test, w, b)
    print(f'Test score: {test_score}')
    sio.savemat('res/logistic_regression.mat', {'w': w, 'b': b, 'train_score': train_score,
                'cv_score': cv_score, 'test_score': test_score, 'best_params': best_params, 'time': end-start})
