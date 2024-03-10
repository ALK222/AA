import copy
import numpy as np
import scipy.io as sio
import utils
import concurrent.futures
import matplotlib.pyplot as plt
import os
import logistic_reg

model_folder = "./models/"


#########################################################################
# one-vs-all
#

def train_model(c, X, y, n_labels, lambda_):
    print(f"Training {c}...")
    initial_theta = np.zeros(X.shape[1] + 1)
    y_i = np.array([1 if label == c else 0 for label in y])
    alpha = 0.2
    lambda_ = 0.01
    num_iters = 25000
    (w, b, _, _) = logistic_reg.gradient_descent(X, y_i, initial_theta[1:], initial_theta[0],
                                                 logistic_reg.compute_cost_reg, logistic_reg.compute_gradient_reg, alpha, num_iters, lambda_)
    print(f"Training {c}... Done")
    return (c, [b] + w.tolist())


def train_all(X, y, n_labels, lambda_):
    all_theta = [0 for _ in range(n_labels)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(train_model, c, X, y, n_labels, lambda_)
                   for c in range(n_labels)]
        for future in concurrent.futures.as_completed(futures):
            c, result = future.result()
            all_theta[c] = result
    return all_theta


def oneVsAll(X, y, n_labels, lambda_):
    """
     Trains n_labels logistic regression classifiers and returns
     each of these classifiers in a matrix all_theta, where the i-th
     row of all_theta corresponds to the classifier for label i.

     Parameters
     ----------
     X : array_like
         The input dataset of shape (m x n). m is the number of
         data points, and n is the number of features. 

     y : array_like
         The data labels. A vector of shape (m, ).

     n_labels : int
         Number of possible labels.

     lambda_ : float
         The logistic regularization parameter.

     Returns
     -------
     all_theta : array_like
         The trained parameters for logistic regression for each class.
         This is a matrix of shape (K x n+1) where K is number of classes
         (ie. `n_labels`) and n is number of features without the bias.
     """
    all_theta = train_all(X, y, n_labels, lambda_)

    return all_theta


def predictOneVsAll(all_theta, X):
    """
    Return a vector of predictions for each example in the matrix X. 
    Note that X contains the examples in rows. all_theta is a matrix where
    the i-th row is a trained logistic regression theta vector for the 
    i-th class. You should set p to a vector of values from 0..K-1 
    (e.g., p = [0, 2, 0, 1] predicts classes 0, 2, 0, 1 for 4 examples) .

    Parameters
    ----------
    all_theta : array_like
        The trained parameters for logistic regression for each class.
        This is a matrix of shape (K x n+1) where K is number of classes
        and n is number of features without the bias.

    X : array_like
        Data points to predict their labels. This is a matrix of shape 
        (m x n) where m is number of data points to predict, and n is number 
        of features without the bias term. Note we add the bias term for X in 
        this function. 

    Returns
    -------
    p : array_like
        The predictions for each data point in X. This is a vector of shape (m, ).
    """
    p = []

    for c in range(len(all_theta)):
        theta = all_theta[c]
        p.append(logistic_reg.function(X, theta[1:], theta[0]))
    p = np.argmax(p, axis=0)
    return p


#########################################################################
# NN
#
def predict(theta1, theta2, X):
    """
    Predict the label of an input given a trained neural network.

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size)

    X : array_like
        The image inputs having shape (number of examples x image dimensions).

    Return 
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """

    a_2 = logistic_reg.function(X, theta1[1:], theta1[0])
    p = logistic_reg.function(a_2, theta2[1:], theta2[0])

    return np.argmax(p, axis=1)


def plot_confusion_matrix(y, p, filename):
    fig, ax = plt.subplots()
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_xticks(np.arange(0, 10))
    ax.set_yticks(np.arange(0, 10))
    ax.set_ylabel("True")
    cm = np.zeros((10, 10))
    for i in range(len(y)):
        cm[y[i] - 1][p[i] - 1] += 1
    cax = ax.matshow(cm, cmap='Reds')
    ax.set_xticks(np.arange(0, 10))
    ax.set_yticks(np.arange(0, 10))
    ax.set_yticks(np.arange(0.5, 10.5), minor='True')
    ax.set_xticks(np.arange(0.5, 10.5), minor='True')
    plt.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)
    fig.colorbar(cax)
    for (i, j), z in np.ndenumerate(cm):
        if i == j:
            ax.text(j, i, '{:0.1f}'.format(z),
                    ha='center', va='center', fontsize=8, color='white')
        else:
            ax.text(j, i, '{:0.1f}'.format(z),
                    ha='center', va='center', fontsize=8)
    plt.savefig(filename, dpi=300)
    plt.clf()


def save_model(all_theta, filename):
    sio.savemat(filename, {'all_theta': all_theta})


def run_one_vs_all():
    data = sio.loadmat('./data/ex3data1.mat', squeeze_me=True)
    X = data['X']
    y = data['y']
    m, n = X.shape
    num_labels = 10
    lambda_ = 0.01
    all_theta = []

    if not os.path.isfile(model_folder + "one_vs_all_train.mat"):
        all_theta = oneVsAll(X, y, num_labels, lambda_)
        save_model(all_theta, model_folder + "one_vs_all_train.mat")
    else:
        aux = sio.loadmat(
            f'{model_folder}one_vs_all_train.mat', squeeze_me=True)
        all_theta = np.array(aux['all_theta'])
        print(all_theta.shape)

    p = predictOneVsAll(all_theta, X)
    print((np.sum(np.array(p) == y) / m) * 100)
    plot_confusion_matrix(y, p, logistic_reg.plot_folder +
                          "confusion_matrix_one_vs_all.png")


def run_neural_network():
    data = sio.loadmat('./data/ex3data1.mat', squeeze_me=True)
    X = data['X']
    y = data['y']
    m, n = X.shape
    weights = sio.loadmat('./data/ex3weights.mat', squeeze_me=True)
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    p = predict(theta1.T, theta2.T, X)

    print((np.sum(np.array(p) == y) / m) * 100)
    plot_confusion_matrix(
        y, p, logistic_reg.plot_folder + "confusion_matrix_nn.png")


def main():
    run_one_vs_all()
    run_neural_network()


if __name__ == "__main__":
    main()
