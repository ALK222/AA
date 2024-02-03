import numpy as np

# def integra_mc_iter(fun, a, b, num_puntos=1000):

#     for i in range(0, num_puntos):

#     print(a,b,sep=" ", end="\n")

def calc_max_func(fun : float,a : int, b :int, num_puntos=1000):
    func_steps : float = (b - a) / num_puntos
    max_func : float = 0 # lo ponemos como el minimo ya que el enunciado dice que usemos funciones con resultado positivo
    for i in np.arange(a,b + func_steps,func_steps):
        max_func = max(max_func, fun(i))
    return max_func


def cuadrado(x : float) -> float:
    return x * x

def main() -> None:
    print(calc_max_func(cuadrado, -5 , 3))

main()