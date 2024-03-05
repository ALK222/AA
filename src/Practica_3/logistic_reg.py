from time import sleep
import numpy as np
import utils
import matplotlib.pyplot as plt
import public_tests
import copy
import math

plot_folder: str = './memoria/imagenes/'
csv_folder: str = './memoria/csv/'
ex_folder: str = './data/'


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """

    g = 1/(1+np.exp(-z))

    return g


#########################################################################
# logistic regression
#

def function_no_sigmoid(x, w, b):
    return np.dot(x, w) + b


def function(x, w, b):
    return sigmoid(np.dot(x, w) + b)


def loss(X, Y, fun, w, b) -> float:
    return (-Y * np.log(fun(X, w, b))) - ((1 - Y) * np.log(1 - fun(X, w, b)))


def compute_cost(X, y, w, b, lambda_=None):
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

    loss_vec = np.vectorize(loss)
    total_cost = 0
    for i in range(X.shape[0]):
        total_cost += loss(X[i], y[i], function, w, b)
    total_cost /= X.shape[0]

    return total_cost


def compute_gradient(X, y, w, b, lambda_=None):
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

    dj_dw = 0
    for i in range(X.shape[0]):
        dj_dw += (function(X[i], w, b) - y[i]) * X[i]
    dj_dw /= X.shape[0]
    dj_db = 0
    for i in range(X.shape[0]):
        dj_db += (function(X[i], w, b) - y[i])
    dj_db /= X.shape[0]

    return dj_db, dj_dw


#########################################################################
# regularized logistic regression
#
def compute_cost_reg(X, y, w, b, lambda_=1):
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


def compute_gradient_reg(X, y, w, b, lambda_=1):
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


#########################################################################
# gradient descent
#
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_=None):
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

    J_history = [cost_function(X, y, w, b, lambda_)]

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b, lambda_)
        w = w - (alpha * dj_dw)
        b -= alpha * dj_db
        J_history.append(cost_function(X, y, w, b, lambda_))
        # print(f'w: {w} b: {b} J: {J_history[-1]}')

    return w, b, np.array(J_history)


#########################################################################
# predict
#
def predict(X, w, b):
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


def loadData(filename):
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 0:2]
    Z = data[:, 2]
    return X, Z


def plot_ex_data():
    (X, Z) = loadData(f'{ex_folder}ex2data1.txt')

    utils.plot_data(X, Z)
    plt.legend(['Admitted', 'Not Admitted'], loc='upper right')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    plt.savefig(f'{plot_folder}muestreo1.png', dpi=300)

    plt.clf()

    (X2, Z2) = loadData(f'{ex_folder}ex2data2.txt')

    utils.plot_data(X2, Z2)
    plt.legend(['Accepted', 'Rejected'], loc='upper right')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    plt.savefig(f'{plot_folder}muestreo2.png', dpi=300)


def run_test():
    test_methods = [
        (public_tests.sigmoid_test, sigmoid),
        (public_tests.compute_cost_test, compute_cost),
        (public_tests.compute_gradient_test, compute_gradient),
        (public_tests.predict_test, predict),
        (public_tests.compute_cost_reg_test, compute_cost_reg),
        (public_tests.compute_gradient_reg_test, compute_gradient_reg)
    ]
    for test, method in test_methods:
        print('\033[37m')
        print(f'Running test: {test.__name__}')
        test(method)


def run_student_sim():
    (X, Z) = loadData(f'{ex_folder}ex2data1.txt')
    w = np.zeros(X.shape[1])
    b = -8
    alpha = 0.001
    num_iters = 10000
    w, b, J_history = gradient_descent(
        X, Z, w, b, compute_cost, compute_gradient, alpha, num_iters)
    plt.clf()
    utils.plot_decision_boundary(w, b, X, Z)
    print(f'Cost: {J_history[-1]}')
    print(f'Predictions: {predict(X, w, b)}')
    plt.savefig(f'{plot_folder}muestreo1_sim.png', dpi=300)
    plt.clf()


def run_chip_sim():
    (X, Z) = loadData(f'{ex_folder}ex2data2.txt')
    X = utils.map_feature(X1=X[:, 0], X2=X[:, 1])
    w = np.zeros(X.shape[1])
    b = 1
    alpha = 0.01
    lambda_ = 0.01
    num_iters = 10000
    w, b, J_history = gradient_descent(
        X, Z, w, b, compute_cost_reg, compute_gradient_reg, alpha, num_iters, lambda_)
    utils.plot_decision_boundary(w, b, X, Z)
    print(f'Cost: {J_history[-1]}')
    print(f'Predictions: {predict(X, w, b)}')
    plt.savefig(f'{plot_folder}muestreo2_sim.png', dpi=300)
    plt.clf()


def main():

    plot_ex_data()
    run_test()

    run_student_sim()
    run_chip_sim()


if __name__ == "__main__":
    main()
