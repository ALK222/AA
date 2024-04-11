import numpy as np
import sys
import os
from commandline import CommandLine
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import sklearn.linear_model as lm
from sklearn import svm
from sklearn import preprocessing
import traceback


class Design:

    model_folder = "./models"
    plot_folder = "./memoria/plots"
    RANDOM_STATE = 1

    def __init__(self):
        self.commandLine: CommandLine = CommandLine()
        self.commandLine.parse(sys.argv[1:])

    def gen_data(self, m: int, seed: int = 1, scale: float = 0.7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate a data set based on a x^2 with added  noise

        Args:
            m (int): size of the data set
            seed (int, optional): seed for the random generator. Defaults to 1.
            scale (float, optional): scale of the noise. Defaults to 0.7.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: x and y train, x and y ideals.
        """
        c = 0
        x_train = np.linspace(0, 49, m)
        np.random.seed(seed)
        y_ideal = x_train**2 + c
        y_train = y_ideal + scale * y_ideal * (np.random.sample((m,)) - 0.5)
        x_ideal = x_train
        return x_train, y_train, x_ideal, y_ideal

    def prep_env(self):
        if (not os.path.exists(self.model_folder)):
            os.makedirs(self.model_folder)
        if (not os.path.exists(self.plot_folder)):
            os.makedirs(self.plot_folder)

    def plot_lines(self, x, y, x_ideal, y_ideal, model_range, model, name: str):
        plt.plot(x, y, 'o', label='Train data')
        plt.plot(x_ideal, y_ideal, label='Ideal data')
        plt.plot(model_range, model, label='Predictions')
        plt.legend()
        plt.savefig(f'{self.plot_folder}/{name}.png', dpi=300)
        plt.clf()

    def create_dataset(self):
        def gen():
            x_train, y_train, x_ideal, y_ideal = self.gen_data(64)
            sio.savemat(f'{self.model_folder}/dataset.mat', {
                        'x_train': x_train, 'y_train': y_train, 'x_ideal': x_ideal, 'y_ideal': y_ideal})
            plt.plot(x_train, y_train, 'o', label='Train data')
            plt.plot(x_ideal, y_ideal, label='Ideal data')
            plt.legend()
            plt.savefig(f'{self.plot_folder}/dataset.png')
        if (not os.path.exists(f'{self.plot_folder}/dataset.png')):
            gen()
        else:

            if (self.commandLine.interactive):
                answer = ''
                while (answer != 'y' and answer != 'n'):
                    print("Recreate dataset? [y/n]")
                    answer = input()
                if (answer == 'y'):
                    gen()

    def train(self, x, y, grado):
        poly = PolynomialFeatures(degree=grado, include_bias=False)
        x_train = poly.fit_transform(x[:, None])
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        model = lm.LinearRegression().fit(x_train, y)
        return poly, scaler, model, x_train

    def train_reg(self, x, y, grado, alpha):
        poly = PolynomialFeatures(degree=grado, include_bias=False)
        x_train = poly.fit_transform(x[:, None])
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        model = lm.Ridge(alpha=alpha).fit(x_train, y)
        return poly, scaler, model, x_train

    def cost(self, x, y):
        m = len(y)
        sum = (np.sum((x - y)**2))
        return sum / (2 * m)

    def test_sobreajuste(self):

        x, y, x_ideal, y_ideal = self.gen_data(64)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=0.33, random_state=self.RANDOM_STATE)

        poly, scal, lin, x_train = self.train(x_train, y_train, 15)

        range_x = np.linspace(np.min(x), np.max(x), 1000)
        range_x = range_x[:, None]
        range_x_p = poly.transform(range_x)
        range_x_p = scal.transform(range_x_p)
        y_pred = lin.predict(range_x_p)

        self.plot_lines(x, y, x_ideal, y_ideal, range_x,
                        y_pred, 'overfitting')

        x_test = poly.transform(x_test[:, None])
        x_test = scal.transform(x_test)
        pred_test = lin.predict(x_test)
        cost_test = self.cost(pred_test, y_test)

        pred_train = lin.predict(x_train)
        cost_train = self.cost(pred_train, y_train)

        print(f"Cost train: {cost_train}")
        print(f"Cost test: {cost_test}")

    def seleccion_grado(self):
        x, y, x_ideal, y_ideal = self.gen_data(64)
        grados = np.arange(1, 11)

        grado_el = 0

        x_train, x_val, y_train, y_val = train_test_split(
            x, y, train_size=0.6, random_state=self.RANDOM_STATE)
        x_cv, x_test, y_cv, y_test = train_test_split(
            x_val, y_val, train_size=0.5, random_state=self.RANDOM_STATE)

        min: np.ndarray = np.empty(shape=(grados.size,))
        minGrado = 1000

        clf_array = np.empty(shape=(grados.size,), dtype=lm.LinearRegression)
        for grado in grados:
            pol, scal, lin, x_train_g = self.train(x_train, y_train, grado)
            clf_array[grado - 1] = lin
            x_cv_g = pol.transform(x_cv[:, None])
            x_cv_g = scal.transform(x_cv_g)
            pred = clf_array[grado - 1].predict(x_cv_g)
            min = np.append(min, self.cost(pred, y_cv))
            if (min[grado] != 0 and minGrado > min[grado]):
                minGrado = min[grado]
                grado_el = grado

            print(f"Grado: {grado} - Score: {min[grado]}")
        opt_grad = grado_el
        print(f"Optimal grado: {opt_grad}")
        X = PolynomialFeatures(
            degree=opt_grad, include_bias=False).fit_transform(x[:, None])
        X = StandardScaler().fit_transform(X)

        self.plot_lines(x, y, x_ideal, y_ideal, x,
                        clf_array[opt_grad - 1].predict(X), 'grado')

    def regularizacion(self):
        x, y, x_ideal, y_ideal = self.gen_data(64)
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, train_size=0.6, random_state=self.RANDOM_STATE)
        x_test, x_cv, y_test, y_cv = train_test_split(
            x_val, y_val, train_size=0.5, random_state=self.RANDOM_STATE)
        grado = 15
        lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
                   1e-1, 1, 10, 100, 300, 600, 900]

        X_train = PolynomialFeatures(
            degree=grado, include_bias=False).fit_transform(x_train[:, None])
        X_train = StandardScaler().fit_transform(X_train)

        X_val = PolynomialFeatures(
            degree=grado, include_bias=False).fit_transform(x_cv[:, None])
        X_val = StandardScaler().fit_transform(X_val)

        model_array = np.empty(shape=(len(lambdas),), dtype=lm.Ridge)

        cost = np.empty(shape=(len(lambdas),))

        for l in lambdas:
            model = lm.Ridge(random_state=self.RANDOM_STATE,
                             alpha=l).fit(X_train, y_train)
            model_array[lambdas.index(l)] = model
            pred = model_array[lambdas.index(l)].predict(X_val)
            cost[lambdas.index(l)] = self.cost(pred, y_cv)
            print(f"Lambda: {l} - Cost: {cost[lambdas.index(l)]}")
        opt_l = np.argmin(np.abs(cost))
        print(opt_l)
        plt.plot(x, y, 'o', label='Train data')
        plt.plot(x_ideal, y_ideal, label='Ideal data')
        plt.plot(x, model_array[opt_l].predict(
            StandardScaler().fit_transform(PolynomialFeatures(degree=grado, include_bias=False).fit_transform(x[:, None]))), label='Predictions')
        plt.legend()
        plt.savefig(f'{self.plot_folder}/regularizacion.png', dpi=300)

    def test_overfitting(self):
        if (not os.path.exists(f'{self.plot_folder}/overfitting.png')):
            self.test_sobreajuste()
        else:
            if (self.commandLine.interactive):
                answer = ''
                while (answer != 'y' and answer != 'n'):
                    print("Recreate overfitting test? [y/n]")
                    answer = input()
                if (answer == 'y'):
                    self.test_sobreajuste()

    def test_seleccion_grado(self):
        if (not os.path.exists(f'{self.plot_folder}/grado.png')):
            self.seleccion_grado()
        else:
            if (self.commandLine.interactive):
                answer = ''
                while (answer != 'y' and answer != 'n'):
                    print("Recreate grado test? [y/n]")
                    answer = input()
                if (answer == 'y'):
                    self.seleccion_grado()

    def test_regularizacion(self):
        if (not os.path.exists(f'{self.plot_folder}/regularizacion.png')):
            self.regularizacion()
        else:
            if (self.commandLine.interactive):
                answer = ''
                while (answer != 'y' and answer != 'n'):
                    print("Recreate regularizacion test? [y/n]")
                    answer = input()
                if (answer == 'y'):
                    self.regularizacion()

    def launch(self):

        funcs = [self.prep_env, self.create_dataset, self.test_overfitting,
                 self.test_seleccion_grado, self.test_regularizacion]

        for func in funcs:
            plt.clf()
            try:
                func()
            except Exception as e:
                print(f"Error: {traceback.format_exc()}")


if __name__ == "__main__":
    design: Design = Design()

    design.launch()
