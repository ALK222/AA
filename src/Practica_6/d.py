import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as sp
import sklearn.model_selection as ms

plot_folder = "./memoria/plots"
RANDOM_STATE = 1


def plot_linear_data(x, y, x_ideal, y_ideal, model_range, model, name: str):
    plt.plot(x, y, 'o', label='Train data')
    plt.plot(x_ideal, y_ideal, label='Ideal data')
    plt.plot(model_range, model, label='Predictions')
    plt.legend()
    plt.savefig(f'{plot_folder}/{name}.png', dpi=300)
    plt.clf()


def draw_learning_curve(x, y, x_i, y_i):
    plt.figure()
    plt.plot(x_i, y_i, c='orange', label='validation')
    plt.plot(x, y, c='blue', label='train')
    plt.legend()
    plt.savefig(f'{plot_folder}/learning_curve.png', dpi=300)


def gen_data(m, seed=1, scale=0.7):
    c = 0
    x_train = np.linspace(0, 49, m)
    np.random.seed(seed)
    y_ideal = x_train**2 + c
    y_train = y_ideal + scale * y_ideal * (np.random.sample((m,)) - 0.5)
    x_ideal = x_train
    return x_train, y_train, x_ideal, y_ideal


def cost(y, y_hat):
    return np.sum((y - y_hat)**2) / y.shape[0]


def train(x_train, y_train, grado):
    poly = sp.PolynomialFeatures(degree=grado, include_bias=False)
    x_train = poly.fit_transform(x_train[:, None])
    scal = sp.StandardScaler()
    x_train = scal.fit_transform(x_train)
    model = lm.LinearRegression()
    model.fit(x_train, y_train)
    return poly, scal, model, x_train


def train_reg(x_train, y_train, grado, l):
    poly = sp.PolynomialFeatures(degree=grado, include_bias=False)
    x_train = poly.fit_transform(x_train[:, None])
    scal = sp.StandardScaler()
    x_train = scal.fit_transform(x_train)
    model = lm.Ridge(alpha=l)
    model.fit(x_train, y_train)
    return poly, scal, model, x_train


def test(x_test, y_test, x_train, y_train, poly, scal, model):
    x_test = poly.transform(x_test[:, None])
    x_test = scal.transform(x_test)

    y_pred_test = model.predict(x_test)
    test_cost = cost(y_pred_test, y_test)
    y_pred_train = model.predict(x_train)
    train_cost = cost(y_pred_train, y_train)
    return test_cost, train_cost


def overfitting(x, y, x_i, y_i):
    x = x
    x_train, x_test, y_train, y_test = ms.train_test_split(
        x, y, test_size=0.33, random_state=RANDOM_STATE)
    pol, scal, model, x_train = train(x_train, y_train, 15)
    range_x = np.linspace(np.min(x), np.max(x), 1000)
    range_x = range_x[:, None]
    range_x_p = pol.transform(range_x)
    range_x_p = scal.transform(range_x_p)
    y_pred = model.predict(range_x_p)
    plot_linear_data(x, y, x_i, y_i, range_x, y_pred, 'overfitting')
    test_cost, train_cost = test(
        x_test, y_test, x_train, y_train, pol, scal, model)

    print(f"Train cost: {train_cost}")
    print(f"Test cost: {test_cost}")


def main():
    x, y, x_i, y_i = gen_data(64)
    overfitting(x, y, x_i, y_i)


if __name__ == '__main__':
    main()
