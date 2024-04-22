from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from ComplexModel import ComplexModel
from SimpleModel import SimpleModel
import torch
import commandline
import os
import sys

plot_folder = './memoria/images'

# Slice the Paired colormap into two segments for each dataset
paired_cmap = plt.cm.get_cmap('Paired')
# "Pastel" colors
cmap_dataset1 = ListedColormap(
    [paired_cmap(2*i) for i in range(6)])
# "Vibrant" colors
cmap_dataset2 = ListedColormap(
    [paired_cmap(2*i+1) for i in range(6)])


def generate_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates an artificial set of data with 6 classes

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: X, y, centers of each class
    """
    classes: int = 6
    m: int = 800
    std: float = 0.4
    center: np.ndarray = np.array(
        [[-1, 0], [1, 0], [0, 1], [0, -1], [-2, 1], [-2, -1]])

    X, y = make_blobs(n_samples=m, centers=center,
                      cluster_std=std, random_state=2, n_features=2)
    return X, y, center


def train_split(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits the data into training, cross validation and test sets
    Args:
        X (np.ndarray): Data
        y (np.ndarray): Labels
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_cv, X_test, y_train, y_cv, y_test
    """
    X_train, X_, y_train, y_ = train_test_split(
        X, y, test_size=0.50, random_state=1)
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_, y_, test_size=0.20, random_state=1)
    return X_train, X_cv, X_test, y_train, y_cv, y_test


def train_data(X_train: np.ndarray, y_train: np.ndarray) -> torch.utils.data.DataLoader:
    """Makes a DataLoader from the training data

    Args:
        X_train (np.ndarray): X train data
        y_train (np.ndarray): targets

    Returns:
        torch.utils.data.DataLoader: Data loader
    """
    X_train_norm: np.ndarray = (X_train - np.mean(X_train)) / np.std(X_train)
    X_train_norm: torch.Tensor = torch.from_numpy(X_train_norm).float()
    y_train: torch.Tensor = torch.from_numpy(y_train)

    train_ds: torch.utils.data.TensorDataset = torch.utils.data.TensorDataset(
        X_train_norm, y_train)

    torch.manual_seed(1)
    batch_size = 2
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size)

    return train_dl


def acc(output: torch.Tensor, target: torch.Tensor) -> float:
    """Tests the accuracy of one prediction

    Args:
        output (torch.Tensor): predicted value
        target (torch.Tensor): target value

    Returns:
        float: accuracy
    """
    return (torch.argmax(output, dim=1) == target).float().sum()


def train_model(model: torch.nn.Sequential, train_dl: torch.utils.data.DataLoader, lossFunction: torch.nn.modules.loss._Loss, optimizer: torch.optim.Optimizer, num_epochs: int) -> tuple[torch.nn.Sequential, np.ndarray, np.ndarray]:
    """Trains the model
    Args:
        model (torch.nn.Sequential): Model to train
        train_dl (torch.utils.data.DataLoader): DataLoader
        lossFunction (torch.nn.modules.loss._Loss): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        num_epochs (int): Number of epochs
    Returns:
        tuple[torch.nn.Sequential, np.ndarray, np.ndarray]: model, loss_hist, accuracy_hist
    """
    log_epocs: int = num_epochs / 100
    loss_hist: np.ndarray = np.zeros(num_epochs)
    accuracy_hist: np.ndarray = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            # Generate output with the model
            outputs: torch.Tensor = model(x_batch)
            # Calculate loss
            loss: float = lossFunction(outputs, y_batch)
            # Reset the gradient
            optimizer.zero_grad()
            # Calculate the gradient
            loss.backward()
            # Update the weights
            optimizer.step()

            loss_hist[epoch] += loss.item() * y_batch.size(0)
            accuracy_hist[epoch] += acc(outputs, y_batch)

        loss_hist[epoch] /= len(train_dl.dataset)
        accuracy_hist[epoch] /= len(train_dl.dataset)
        if epoch % log_epocs == 0:
            print(
                f"Epoch {epoch} Loss {loss_hist[epoch]} Accuracy {accuracy_hist[epoch]}")
    return model, loss_hist, accuracy_hist


def complex_model(X_train: np.ndarray, y_train: np.ndarray) -> tuple[torch.nn.Sequential, np.ndarray, np.ndarray]:
    """This complex model uses a 3 layer neural network to classify the data:
        - Dense layer with 120 units and relu activation function
        - Dense layer with 40 units and relu activation function
        - Dense layer with 6 units and linear activation function

    Args:
        X_train (np.ndarray): Training data
        y_train (np.ndarray): Training labels
    Returns:
        tuple[torch.nn.Sequential, np.ndarray, np.ndarray]: model, loss_hist, accuracy_hist
    """
    lossFunction = torch.nn.CrossEntropyLoss()
    num_epochs = 1000
    learning_rate = 0.001
    model = ComplexModel()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)
    train_dl = train_data(X_train, y_train)

    return train_model(model, train_dl, lossFunction, optimizer, num_epochs)


def simple_model(X_train: np.ndarray, y_train: np.ndarray) -> tuple[torch.nn.Sequential, np.ndarray, np.ndarray]:
    """This simple model uses a 2 layer neural network to classify the data:
        - Dense layer with 6 units and relu activation function
        - Dense layer with 6 units and linear activation function
    Args:
        X_train (np.ndarray): Training data
        y_train (np.ndarray): Training labels
    Returns:
        tuple[torch.nn.Sequential, np.ndarray, np.ndarray]: model, loss_hist, accuracy_hist
    """

    epochs = 1000
    learning_rate = 0.01
    model = SimpleModel()
    lossFunction = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dl = train_data(X_train, y_train)

    return train_model(model, train_dl, lossFunction, optimizer, epochs)


def regularized_model(X_train: np.ndarray, y_train: np.ndarray, _lambda: float) -> tuple[torch.nn.Sequential, np.ndarray, np.ndarray]:
    """This regularized model uses a 3 layer neural network to classify the data:
        - Dense layer with 120 units and relu activation function
        - Dense layer with 40 units and relu activation function
        - Dense layer with 6 units and linear activation function

    Args:
        X_train (np.ndarray): Training data
        y_train (np.ndarray): Training labels
    Returns:
        tuple[torch.nn.Sequential, np.ndarray, np.ndarray]: model, loss_hist, accuracy_hist
    """
    lossFunction = torch.nn.CrossEntropyLoss()
    num_epochs = 1000
    learning_rate = 0.001
    model = ComplexModel()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=_lambda)
    train_dl = train_data(X_train, y_train)

    return train_model(model, train_dl, lossFunction, optimizer, num_epochs)


def plot_loss_accuracy(loss_hist: np.ndarray, accuracy_hist: np.ndarray, name: str) -> None:
    """Plots the loss and accuracy history of a given model

    Args:
        loss_hist (np.ndarray): loss history over the epochs
        accuracy_hist (np.ndarray): accuracy history over the epochs
        name (str): name of the file inside the plot folder
    """
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(loss_hist)
    ax.set_title('Training loss', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(accuracy_hist)
    ax.set_title('Training accuracy', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig(f'{plot_folder}/{name}.png', dpi=300)
    plt.clf()


def plot_decision_boundary(X_train: np.ndarray, Y_train: np.ndarray, X_cv: np.ndarray, Y_cv: np.ndarray, model: torch.nn.Sequential, name: str) -> None:
    """Plots the decision boundary of a given model
    Args:
        X_train (np.ndarray): Training data
        Y_train (np.ndarray): Training labels
        X_cv (np.ndarray): Cross validation data
        Y_cv (np.ndarray): Cross validation labels
        model (torch.nn.Sequential): Model
        name (str): name of the file inside the plot folder
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100),
                         np.linspace(-3, 3, 100))

    Z = model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())
    Z = torch.argmax(Z, dim=1).reshape(xx.shape)

    for ax, X, Y, title, cmap_dataset in zip(axes, [X_train, X_cv], [Y_train, Y_cv], ['Training data', 'Cross validation data'], [cmap_dataset2, cmap_dataset1]):
        ax.set_title(title, size=15)
        ax.set_xlabel('X0', size=15)
        ax.set_ylabel('X1', size=15)

        ax.contour(xx, yy, Z, alpha=1, colors=['darkgreen'], linewidths=[2])
        ax.scatter(X[:, 0], X[:, 1], c=Y, marker=".", cmap=cmap_dataset)

    plt.tight_layout()
    plt.savefig(f'{plot_folder}/{name}.png', dpi=300)
    plt.clf()


def evaluate_model(X_train: np.ndarray, y_train: np.ndarray, X_cv: np.ndarray, y_cv: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, model: torch.nn.Sequential) -> dict[str, float]:
    """Evaluates the model with the training, cross validation and test data
    Args:
        X_train (np.ndarray): Training data
        y_train (np.ndarray): Training labels
        X_cv (np.ndarray): Cross validation data
        y_cv (np.ndarray): Cross validation labels
        X_test (np.ndarray): Test data
        y_test (np.ndarray): Test labels
        model (torch.nn.Sequential): Model
    Returns:
        dict[str, float]: dictionary with the accuracy of the training, cross validation and test data
    """

    data = {'train': (X_train, y_train), 'cv': (
        X_cv, y_cv), 'test': (X_test, y_test)}
    res = {}
    for key, value in data.items():
        X, y = value
        X_norm = (X - np.mean(X)) / np.std(X)
        X_norm = torch.from_numpy(X_norm).float()
        y = torch.from_numpy(y)
        pred = model(X_norm)
        acc = (torch.argmax(pred, dim=1) == y).float().mean()
        print(f"{key} accuracy: {acc:.4f}")
        res[key] = acc

    return res


def plot_data(X_train: np.ndarray, y_train: np.ndarray, X_cv: np.ndarray, y_cv: np.ndarray, centers: np.ndarray, radius: float, name: str) -> None:
    """Plots the data with the training and cross validation data
    Args:
        X_train (np.ndarray): Training data
        y_train (np.ndarray): Training labels
        X_cv (np.ndarray): Cross validation data
        y_cv (np.ndarray): Cross validation labels
        centers (np.ndarray): centers of the classes
        radius (float): radius of the classes
        name (str): name of the file inside the plot folder
    """

    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=y_train, marker=".", cmap=cmap_dataset2)
    plt.scatter(X_cv[:, 0], X_cv[:, 1], c=y_cv, marker='<', cmap=cmap_dataset1)
    circles = [plt.Circle(centers[i], radius * 2, color=cmap_dataset2(i), fill=False)
               for i in range(6)]
    for circle in circles:
        plt.gca().add_artist(circle)

    plt.savefig(f'{plot_folder}/{name}.png', dpi=300)


def plot_regularization(train_error: np.ndarray, cv_error, labmbda_hist: np.ndarray, name: str) -> None:
    """Plots the regularization error
    Args:
        train_error (np.ndarray): Training error
        labmbda_hist (np.ndarray): Lambda history
        name (str): name of the file inside the plot folder
    """
    plt.plot(labmbda_hist, train_error, marker='o', linestyle='-',
             color='blue', label='Train error')
    plt.plot(labmbda_hist, cv_error, marker='o', linestyle='-',
             color='orange', label='CV error')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.title('Regularization error')
    plt.savefig(f'{plot_folder}/{name}.png', dpi=300)
    plt.clf()


def generate_data_driver(commandLine: commandline.CommandLine) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates the data and splits it into training, cross validation and test sets
    Args:
        commandLine (commandline.CommandLine): command line arguments
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_cv, X_test, y_train, y_cv, y_test
    """
    plt.clf()
    X, y, center = generate_data()
    X_train, X_cv, X_test, y_train, y_cv, y_test = train_split(X, y)

    if not os.path.exists(f'{plot_folder}/dataset.png'):
        plot_data(X_train, y_train, X_cv, y_cv, center, 0.4, 'dataset')
    elif commandLine.plot or commandLine.all:
        plot_data(X_train, y_train, X_cv, y_cv, center, 0.4, 'dataset')
    return X_train, X_cv, X_test, y_train, y_cv, y_test


def simple_model_driver(X_train: np.ndarray, y_train: np.ndarray, X_cv: np.ndarray, y_cv: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, commandLine: commandline.CommandLine) -> None:
    """Driver for the simple model
    Args:
        X_train (np.ndarray): Training data
        y_train (np.ndarray): Training labels
        X_cv (np.ndarray): Cross validation data
        y_cv (np.ndarray): Cross validation labels
        X_test (np.ndarray): Test data
        y_test (np.ndarray): Test labels
        commandLine (commandline.CommandLine): command line arguments
    """
    plt.clf()
    if not os.path.exists(f'{plot_folder}/loss_accuracy_simple.png') or commandLine.all or commandLine.simple:
        model, loss_hist, accuracy_hist = simple_model(
            X_train, y_train)
        plot_loss_accuracy(loss_hist, accuracy_hist, 'loss_accuracy_simple')
        evaluate_model(X_train, y_train, X_cv, y_cv, X_test, y_test, model)
        plt.title('Simple model')
        plot_decision_boundary(X_train, y_train, X_cv, y_cv, model,
                               'decision_boundary_simple')


def complex_model_driver(X_train: np.ndarray, y_train: np.ndarray, X_cv: np.ndarray, y_cv: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, commandLine: commandline.CommandLine) -> None:
    """Driver for the complex model
    Args:
        X_train (np.ndarray): Training data
        y_train (np.ndarray): Training labels
        X_cv (np.ndarray): Cross validation data
        y_cv (np.ndarray): Cross validation labels
        X_test (np.ndarray): Test data
        y_test (np.ndarray): Test labels
        commandLine (commandline.CommandLine): command line arguments
    """
    plt.clf()
    if not os.path.exists(f'{plot_folder}/loss_accuracy_complex.png') or commandLine.all or commandLine.complex:
        model, loss_hist, accuracy_hist = complex_model(
            X_train, y_train)
        plot_loss_accuracy(loss_hist, accuracy_hist, 'loss_accuracy_complex')
        evaluate_model(X_train, y_train, X_cv, y_cv, X_test, y_test, model)
        plt.title('Complex model')
        plot_decision_boundary(X_train, y_train, X_cv, y_cv, model,
                               'decision_boundary_complex')


def regularized_model_driver(X_train: np.ndarray, y_train: np.ndarray, X_cv: np.ndarray, y_cv: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, commandLine: commandline.CommandLine) -> None:
    """Driver for the regularized model
    Args:
        X_train (np.ndarray): Training data
        y_train (np.ndarray): Training labels
        X_cv (np.ndarray): Cross validation data
        y_cv (np.ndarray): Cross validation labels
        X_test (np.ndarray): Test data
        y_test (np.ndarray): Test labels
        commandLine (commandline.CommandLine): command line arguments
    """
    plt.clf()
    if not os.path.exists(f'{plot_folder}/loss_accuracy_regularized.png') or commandLine.all or commandLine.regularized:
        model, loss_hist, accuracy_hist = regularized_model(
            X_train, y_train, 0.1)
        plot_loss_accuracy(loss_hist, accuracy_hist,
                           'loss_accuracy_regularized')
        evaluate_model(X_train, y_train, X_cv, y_cv, X_test, y_test, model)
        plt.title('Regularized model')
        plot_decision_boundary(X_train, y_train, X_cv, y_cv, model,
                               'decision_boundary_regularized')


def reg_test_driver(X_train: np.ndarray, y_train: np.ndarray, X_cv: np.ndarray, y_cv: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, commandLine: commandline.CommandLine) -> None:
    plt.clf()
    if not os.path.exists(f'{plot_folder}/tuning.png') or commandLine.all or commandLine.iter:
        _lambda_hist = np.array([0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3])
        error_hist_train = np.empty_like(_lambda_hist)
        error_hist_cv = np.empty_like(_lambda_hist)
        for i in range(len(_lambda_hist)):
            model, _, _ = regularized_model(
                X_train, y_train, _lambda_hist[i])
            res = evaluate_model(X_train, y_train, X_cv,
                                 y_cv, X_test, y_test, model)
            error_hist_train[i] = res['train']
            error_hist_cv[i] = res['cv']

        plot_regularization(error_hist_train, error_hist_cv,
                            _lambda_hist, 'tuning')


def main():
    commandLine = commandline.CommandLine()
    commandLine.parse(sys.argv[1:])
    X_train, X_cv, X_test, y_train, y_cv, y_test = generate_data_driver(
        commandLine)

    complex_model_driver(X_train, y_train, X_cv, y_cv,
                         X_test, y_test, commandLine)
    simple_model_driver(X_train, y_train, X_cv, y_cv,
                        X_test, y_test, commandLine)
    regularized_model_driver(X_train, y_train, X_cv, y_cv,
                             X_test, y_test, commandLine)
    reg_test_driver(X_train, y_train, X_cv, y_cv, X_test, y_test, commandLine)


if __name__ == "__main__":
    main()
