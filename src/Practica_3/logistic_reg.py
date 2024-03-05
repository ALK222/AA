import numpy as np
import utils
import matplotlib.pyplot as plt
import public_tests
import copy

plot_folder: str = './memoria/imagenes/'
csv_folder: str = './memoria/csv/'
ex_folder: str = './data/'


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


#########################################################################
# logistic regression
#
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
    return (-Y * np.log(fun(X, w, b))) - ((1 - Y) * np.log(1 - fun(X, w, b)))


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

    func = function(X, w,b)
    
    dj_dw = np.dot(func -y, X)
    dj_dw /= X.shape[0]

    dj_db = np.sum(func - y)
    dj_db /= X.shape[0]

    return dj_db, dj_dw


#########################################################################
# regularized logistic regression
#
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


#########################################################################
# gradient descent
#
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


#########################################################################
# predict
#
def predict(X, w, b)->np.ndarray:
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


def load_data(filename:str) -> tuple[np.ndarray, np.ndarray]:
    """Loads the train data

    Args:
        filename (str): dataset filename

    Returns:
        tuple[np.ndarray, np.ndarray]: Train data and expected values
    """
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 0:2]
    Z = data[:, 2]
    return X, Z


def plot_ex_data() -> None:
    """Plots the given datasets into a graph
    """
    (X, Z) = load_data(f'{ex_folder}ex2data1.txt')

    utils.plot_data(X, Z)
    plt.legend(['Admitted', 'Not Admitted'], loc='upper right')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    plt.savefig(f'{plot_folder}muestreo1.png', dpi=300)

    plt.clf()

    (X2, Z2) = load_data(f'{ex_folder}ex2data2.txt')

    utils.plot_data(X2, Z2)
    plt.legend(['Accepted', 'Rejected'], loc='upper right')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.savefig(f'{plot_folder}muestreo2.png', dpi=300)
    plt.clf()

def plot_linear_data(X: np.ndarray, Y: np.ndarray, filename: str, scale:str = 'linear') -> None:
    """Plots the given data into a graph

    Args:
        X (np.ndarray): X data
        Y (np.ndarray): Y data
        filename (str): file to store the graph
    """
    plt.clf()
    plt.plot(Y, X)
    plt.xscale(scale)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.savefig(f'{plot_folder}{filename}', dpi=300)
    plt.clf()


def run_test() -> None:
    """Runs the given tests
    """
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


def run_student_sim() -> None:
    """Trains the model for the first dataset and plots the results
    """
    (X, Z) = load_data(f'{ex_folder}ex2data1.txt')
    w = np.zeros(X.shape[1])
    b = -8
    alpha = 0.001
    num_iters = 10000
    w, b, J_history, predict_history = gradient_descent(
        X, Z, w, b, compute_cost, compute_gradient, alpha, num_iters)
    plt.clf()
    utils.plot_decision_boundary(w, b, X, Z)
    plt.savefig(f'{plot_folder}muestreo1_sim.png', dpi=300)
    plt.clf()
    plot_linear_data(J_history, range(num_iters + 1), 'muestreo1_cost.png', 'log')
    plot_linear_data(predict_history, range(num_iters + 1), 'muestreo1_accuracy.png', 'log')
    print(f'Cost: {J_history[-1]}, w: {w}, b: {b}')
    print(f'Accuracy rate: {predict_history[-1] * 100}%')


def run_chip_sim() -> None:
    """Trains the model for the second dataset and plots the results
    """
    (X, Z) = load_data(f'{ex_folder}ex2data2.txt')
    X = utils.map_feature(X1=X[:, 0], X2=X[:, 1])
    w = np.zeros(X.shape[1])
    b = 1
    alpha = 0.01
    lambda_ = 0.01
    num_iters = 10000
    w, b, J_history, predict_history = gradient_descent(
        X, Z, w, b, compute_cost_reg, compute_gradient_reg, alpha, num_iters, lambda_)
    utils.plot_decision_boundary(w, b, X, Z)
    plt.savefig(f'{plot_folder}muestreo2_sim.png', dpi=300)
    plt.clf()
    plot_linear_data(J_history, range(num_iters + 1), 'muestreo2_cost.png')
    plot_linear_data(predict_history, range(num_iters + 1), 'muestreo2_accuracy.png')
    print(f'Cost: {J_history[-1]}, w: {w}, b: {b}')
    print(f'Accuracy rate: {predict_history[-1] * 100}%')


def main() -> None:

    plot_ex_data()
    run_test()

    run_student_sim()
    run_chip_sim()


if __name__ == "__main__":
    main()
