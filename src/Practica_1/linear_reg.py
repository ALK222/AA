import numpy as np
import public_tests
import utils
import matplotlib.pyplot as plt
from matplotlib import cm
import csv

dtype = np.float64

#########################################################################
# Cost function
#


def fun(x: float, w: float, b: float) -> float:
    """Slope function

    Args:
        x (float): x coordinate
        w (float): slope
        b (float): starting point

    Returns:
        float: predicted y
    """
    return (w*x)+b


def compute_cost(x: np.ndarray, y: np.ndarray, w: float, b: float) -> float:
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m: int = x.size
    x_fun: np.ndarray = np.vectorize(fun, otypes=[dtype])(x, w, b)
    total_cost = (1.0/(2 * m)) * \
        np.sum((np.subtract(x_fun, y)) ** 2)

    return total_cost


#########################################################################
# Gradient function
#
def compute_gradient(x: np.ndarray, y: np.ndarray, w: float, b: float) -> list[float, float]:
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """

    v_func = np.vectorize(fun, otypes=[dtype])
    x_fun = v_func(x, w, b)
    dj_dw = (1/x.size) * np.sum((x_fun - y)*x)
    dj_db = (1/x.size) * np.sum(x_fun - y)

    return dj_dw, dj_db


#########################################################################
# gradient descent
#
def gradient_descent(x: np.ndarray, y: np.ndarray, w_in: float, b_in: float, cost_function: float, gradient_function: float, alpha: float = 0.01, num_iters: int = 1500) -> list[list[float], list[float], list[float]]:
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar) Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """
    m = x.size
    w = [w_in]
    b = [b_in]
    J_history = [cost_function(x, y, w[-1], b)]
    for i in range(1, num_iters):
        (dj_dw, dj_db) = gradient_function(x, y, w[-1], b[-1])
        total_cost = cost_function(x, y, w[-1], b[-1])

        w.append(w[-1] - (alpha * dj_dw))
        b.append(b[-1] - (alpha * dj_db))

        J_history.append(total_cost)

    return w, b, J_history


def make_grid(w_range: list[float], b_range: list[float], X: np.ndarray, Y: np.ndarray, step: float = 0.1) -> list[np.ndarray]:
    """Makes a grid for future graphs using the given data

    Args:
        w_range (list[float]): range of the w value
        b_range (list[float]): range of the b value
        X (np.ndarray): X values of the given data
        Y (np.ndarray): Y values of the given data
        step (float, optional): steps for each range. Defaults to 0.1.

    Returns:
        list[np.ndarray]: W, B, Cost as 2D arrays
    """
    W = np.arange(w_range[0], w_range[1], step)
    B = np.arange(b_range[0],  b_range[1], step)
    W, B = np.meshgrid(W, B)
    cost = np.empty_like(W)

    for ix, iy in np.ndindex(W.shape):
        cost[ix, iy] = compute_cost(X, Y, W[ix, iy], B[ix, iy])

    return [W, B, cost]


def show_contour(data: np.ndarray, wb: list[float], x: np.ndarray, y: np.ndarray) -> None:
    """Saves the contour graph of the program

    Args:
        data (np.ndarray): W, B, cost
        wb (list[float]): final w and b values
        x (np.ndarray): X values given
        y (np.ndarray): Y values given
    """
    plt.clf()
    plt.contour(data[0], data[1], data[2], np.logspace(-2, 3, 20))
    plt.plot(wb[0], wb[1], 'rx')
    annotate_cost = compute_cost(x, y, wb[0], wb[1])
    plt.annotate(f'{annotate_cost}', (wb[0] + 0.2, wb[1] + 0.2), bbox=dict(
        facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    plt.xlabel('w')
    plt.ylabel('b')
    plt.savefig("./memoria/imagenes/contour_plot.png", dpi=300)
    plt.clf()


def show_mesh(data: np.ndarray) -> None:
    """Show the mesh graph for the data

    Args:
        data (np.ndarray): W, B, Cost
    """
    plt.clf()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(data[0], data[1], data[2], cmap=cm.jet,
                           antialiased=False, linewidth=2)

    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.zaxis.set_rotate_label(False)  # disable automatic rotation
    ax.set_zlabel("J(w,b)", rotation=90)

    ax.view_init(10, 250)
    fig.savefig("./memoria/imagenes/mesh.png", dpi=300)
    plt.clf()


def show_scatter_line(x: np.ndarray, y: np.ndarray, wb: list[float]) -> None:
    """Scatters the given data and plot the prediction line

    Args:
        x (np.ndarray): X given data
        y (np.ndarray): Y given data
        wb (list[float]): final values of w and b
    """
    plt.clf()
    plt.scatter(x, y, color='red', marker='x', label='observados')
    f_vec = np.vectorize(fun)
    y_vec = f_vec(x, wb[0], wb[1])

    # plt.scatter(35, fun(35, wb[0], wb[1]), marker='o', color='green')
    # plt.scatter(70, fun(70, wb[0], wb[1]), marker='o', color='green')

    plt.plot(x, y_vec, color='blue', label='prediccion')

    plt.xlabel('tam. población (1000x)')
    plt.ylabel('precio vivienda (1000$)')

    plt.savefig('./memoria/imagenes/scatter.png', dpi=300)
    plt.clf()


def write_results(J_history: np.ndarray, w: np.ndarray, b: np.ndarray) -> None:
    """Writes simplified results to csv for data visualization

    Args:
        J_history (np.ndarray): Cost history of the linear regression
        w (np.ndarray): w history
        b (np.ndarray): b history
    """
    with open('./memoria/recursos/J_history_simplificado.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteracion', 'J'])
        slicedHistory = J_history[0:-1:1000]
        for i in range(1, len(slicedHistory) + 1):
            writer.writerow([i*1000, slicedHistory[i - 1]])

    with open('./memoria/recursos/predicciones.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Tam_poblacion', 'coste_predecido'])
        writer.writerow([35, fun(35, w[-1], b[-1])])
        writer.writerow([70, fun(70, w[-1], b[-1])])


def show_J_history(J_history: np.ndarray) -> None:
    """Show J_history data in two graphs

    Args:
        J_history (ndarray): J history
    """
    plt.clf()
    plt.plot(range(0, len(J_history)), J_history)

    plt.xlabel('Num. Iteraciones')
    plt.ylabel('J(w, b)')
    plt.xscale('log')

    plt.savefig('./memoria/imagenes/J_history_log.png')
    plt.xscale('linear')
    plt.savefig('./memoria/imagenes/J_history_linear.png')
    plt.clf()


def main():
    public_tests.compute_cost_test(compute_cost)
    public_tests.compute_gradient_test(compute_gradient)

    (x, y) = utils.load_data()

    num_iters = 15000

    (w, b, J_history) = gradient_descent(
        x, y, 0, 0, compute_cost, compute_gradient, num_iters=num_iters)

    grid_data = make_grid([
        -1, 4], [-10, 10], x, y)

    show_contour(grid_data, [w[-1], b[-1]], x, y)
    show_mesh(grid_data)
    show_J_history(J_history)
    show_scatter_line(x, y, [w[-1], b[-1]])

    write_results(J_history, w, b)


main()
