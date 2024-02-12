from io import TextIOWrapper  # abrir y/o crear archivos de datos
import numpy as np  # vectorizaciones
from math import sin, pi  # para la funcion seno y su intervalo
import matplotlib.pyplot as plt  # para hacer los distintos graficos
import random  # para generar puntos aleatorios en las funciones iterativas
import time  # para medir los tiempos de ejecucian
import csv  # para guardar los datos de manera ordenada


def check_point(fun: float, x: float, y: float) -> bool:
    """Checks if a point if above or below the function given

    Args:
        fun (float): function to check
        x (float): x coordinate of the point
        y (float): y coordinate of the point

    Returns:
        bool: True if below, false if above
    """
    return fun(x) > y


def integra_mc_vect(fun: float, a: float, b: float, num_puntos: int = 1000) -> float:
    """Calculates the integral of a function by the montecarlo method. Vectorized way.

    Args:
        fun (float): function to integrate
        a (float): lower bound of the interval
        b (float): upper bound of the interval
        num_puntos (int, optional): number of random points to use. Defaults to 1000.

    Returns:
        float: Value of the calculated integral
    """

    vec_fun = np.vectorize(fun)
    coord_x: ndarray[float] = np.linspace(a, b, num_puntos)
    real_y = vec_fun(coord_x)
    max_func: float = np.max(real_y)

    coord_y: ndarray[float] = np.random.uniform(0, max_func, num_puntos)

    n_debajo = np.sum(coord_y < real_y)

    montecarlo = (n_debajo / num_puntos) * (b-a) * max_func

    return montecarlo


def integra_mc_iter(fun: float, a: float, b: float, num_puntos: int = 1000) -> float:
    """Calculates the integral of a function by the montecarlo method. Iterative way.

    Args:
        fun (float): function to integrate
        a (float): lower bound of the interval
        b (float): upper bound of the interval
        num_puntos (int, optional): number of random points to use. Defaults to 1000.

    Returns:
        float: Value of the calculated integral
    """
    max_func: float = calc_max_func(fun, a, b, num_puntos)

    coord_x: list[float] = [random.uniform(a, b) for _ in range(0, num_puntos)]
    coord_y: list[float] = [random.uniform(
        0, max_func) for _ in range(0, num_puntos)]

    n_debajo: int = 0
    for i in range(0, num_puntos):
        if (check_point(fun, coord_x[i], coord_y[i])):
            n_debajo += 1
    montecarlo: float = (n_debajo / num_puntos) * (b-a) * max_func
    return montecarlo


def calc_max_func(fun: float, a: float, b: float, num_puntos: int = 1000) -> float:
    """Calculates the maximum y value of a function using random points in the x axis

    Args:
        fun (float): function to calculate
        a (float): lower bound of the interval
        b (float): upper bound of the interval
        num_puntos (int, optional): number of random points generated. Defaults to 1000.

    Returns:
        float: max value of the y axis
    """
    func_steps: float = (b - a) / num_puntos
    # lo ponemos como el minimo ya que el enunciado dice que usemos funciones con resultado positivo
    max_func: float = 0

    points: list[float] = [round(random.uniform(a, b+func_steps), num_puntos)
                           for _ in range(0, num_puntos)]
    for i in points:
        max_func = max(max_func, fun(i))
    return max_func


def cuadrado(x: float) -> float:
    """Test function to calculate the square of a number

    Args:
        x (float): number to operate with

    Returns:
        float: x*x
    """
    return x * x


def fun2(x: float) -> float:
    """Test function that calculates a random function

    Args:
        x (float): value to calculate

    Returns:
        float: 1 + x / 10
    """
    return 1 + x / 10


def test_increment(caso_inicial: int, fun: float, a: float, b: float) -> (float, float):
    """Test a set number of cases, incrementing 9 time by the initial case

    Args:
        caso_inicial (int): initial case
        fun (float): function to test
        a (float): lower bound of the interval
        b (float): upper bound of the interval

    Returns:
        float, float: times of the iterative tests, times of the vectorized tests
    """
    total_it = 0
    total_vec = 0

    for _ in range(0, 5):
        # Caso iterativo
        it_start: int = time.process_time()
        sol_it: float = integra_mc_iter(fun, a, b, caso_inicial)
        it_end: int = time.process_time()
        total_it += 1000 * (it_end - it_start)
        print(
            f'caso iterativo {caso_inicial}: {sol_it} -> {1000*(it_end-it_start)}ms')

        # Caso vectorizado
        vec_start = time.process_time()
        sol_vect: float = integra_mc_vect(fun, a, b, caso_inicial)
        vec_end = time.process_time()
        total_vec += 1000 * (vec_end - vec_start)
        print(
            f'caso vectorial {caso_inicial}: {sol_vect} -> {1000*(vec_end - vec_start)}ms')
    print(f'tiempo medio it: {total_it/5}ms')
    print(f'tiempo medio vec: {total_vec/5}ms')
    file: TextIOWrapper = open(
        f"./memoria/recursos/{fun.__name__}.csv", "a")
    writer = csv.writer(file)
    writer.writerow([caso_inicial, round(
        total_it/5, 8), round(total_vec/5, 8)])
    file.close()

    return (total_it/5, total_vec/5)


def gen_fun_graph(fun: float, a: float, b: float) -> None:
    """Generates the graph of a given function on a given interval. Graph also includes some sample points for the montecarlo method

    Args:
        fun (float): function given
        a (float): lower bound of the interval
        b (float): upper bound of the interval
    """
    num_puntos = 100
    step = (b - a) / num_puntos
    func_x = np.arange(a, b + step, step)
    func_y = np.vectorize(fun)(func_x)

    plt.plot(func_x, func_y)

    max_func = np.max(func_y)

    coord_x: ndarray[float] = (b-a) * np.random.random_sample(num_puntos) + a
    coord_y: ndarray[float] = max_func * np.random.random_sample(num_puntos)
    plt.scatter(coord_x, coord_y, color="red", marker="x")
    plt.savefig(f"./memoria/imagenes/{fun.__name__}.png", dpi=300)
    plt.close()


def test_cases() -> None:
    """Battery of tests to generate time data for the report
    """
    fun_list = [cuadrado, sin, fun2]  # functions to test
    upper: list[float] = [5, pi, 100]  # upper bound for each function
    lower: list[float] = [-5, 0, 0]  # lower bound for each function

    # initial test cases
    casos_iniciales: list[int] = np.linspace(10, 10000000, 20)
    print(casos_iniciales)

    for i in range(0, len(fun_list)):
        gen_fun_graph(fun_list[i], lower[i], upper[i])
        tiempo_iter: list[float] = []
        tiempo_vect: list[float] = []
        casos: list[int] = []
        print(f'{fun_list[i].__name__}: [{lower[i]},{upper[i]}]')
        file: TextIOWrapper = open(
            f"./memoria/recursos/{fun_list[i].__name__}.csv", "w")
        writer = csv.writer(file)
        writer.writerow(['', 'Iterativo', 'Vectorizado'])
        file.close()
        for caso in casos_iniciales:
            (tiempo_iter_aux, tiempo_vect_aux) = test_increment(
                int(caso), fun_list[i], lower[i], upper[i])

            tiempo_iter.append(tiempo_iter_aux)
            tiempo_vect.append(tiempo_vect_aux)
            # last case is duplicate so we don't use it
            casos.append(int(caso))

        plt.figure(i, figsize=(8, 5))  # different plot for each function
        plt.scatter(casos, tiempo_iter, label="iterativo")
        plt.scatter(casos, tiempo_vect, label="vectorizado")
        plt.xlabel('Número de puntos')
        plt.ylabel('Tiempo (ms)')
        plt.legend()

        plt.savefig(
            f'./memoria/imagenes/tiempos_{fun_list[i].__name__}.png', dpi=300)
        plt.clf()


def main() -> None:
    test_cases()


if __name__ == "__main__":
    main()
