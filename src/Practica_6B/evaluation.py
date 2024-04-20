from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import commandline
import os
import sys

plot_folder = './memoria/images'

# Slice the Paired colormap into two segments for each dataset
paired_cmap = plt.cm.get_cmap('Paired')
cmap_dataset1 = ListedColormap(
    [paired_cmap(2*i) for i in range(6)])
cmap_dataset2 = ListedColormap(
    [paired_cmap(2*i+1) for i in range(6)])


def generate_data():
    classes: int = 6
    m: int = 800
    std: float = 0.4
    center: np.array = np.array(
        [[-1, 0], [1, 0], [0, 1], [0, -1], [-2, 1], [-2, -1]])

    X, y = make_blobs(n_samples=m, centers=center,
                      cluster_std=std, random_state=2, n_features=2)
    return X, y, center


def train_split(X, y):
    X_train, X_, y_train, y_ = train_test_split(
        X, y, test_size=0.50, random_state=1)
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_, y_, test_size=0.20, random_state=1)
    return X_train, X_cv, X_test, y_train, y_cv, y_test


def train_data(X_train: np.ndarray, y_train: np.ndarray) -> torch.utils.data.DataLoader:
    X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
    X_train_norm = torch.from_numpy(X_train_norm).float()
    y_train = torch.from_numpy(y_train)

    train_ds = torch.utils.data.TensorDataset(X_train_norm, y_train)

    torch.manual_seed(1)
    batch_size = 2
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)

    return train_dl


def complex_model(X_train: np.ndarray, y_train: np.ndarray):
    """This complex model uses a 3 layer neural network to classify the data:
        - Dense layer with 120 units and relu activation function
        - Dense layer with 40 units and relu activation function
        - Dense layer with 6 units and linear activation function

    Args:
        X_train (np.ndarray): Training data
        y_train (np.ndarray): Training labels
    """
    lossFunction = torch.nn.CrossEntropyLoss()
    num_epochs = 1000
    learning_rate = 0.001
    input_size = X_train.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 120),
        torch.nn.ReLU(),
        torch.nn.Linear(120, 40),
        torch.nn.ReLU(),
        torch.nn.Linear(40, 6)
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)
    train_dl = train_data(X_train, y_train)
    log_epochs = num_epochs / 100
    loss_hist = [0] * num_epochs
    accuracy_hist = [0] * num_epochs

    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            # Compute prediction error
            pred = model(x_batch)
            loss = lossFunction(pred, y_batch.long())
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # Compute accuracy
            loss_hist[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) ==
                          y_batch).float()
            accuracy_hist[epoch] += is_correct.sum()
            loss_hist[epoch] /= len(train_dl.dataset)
            accuracy_hist[epoch] /= len(train_dl.dataset)
        if epoch % log_epochs == 0:
            print(f"Epoch {epoch} Loss {loss_hist[epoch]:.10f}")
    print(f"Epoch {1000} Loss {loss_hist[1000 - 1]:.10f}")
    return model, loss_hist, accuracy_hist


def simple_model(X_train: np.ndarray, y_train: np.ndarray):
    input_size = X_train.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 6),
        torch.nn.ReLU(),
        torch.nn.Linear(6, 6)
    )
    epochs = 1000
    lossFunction = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_dl = train_data(X_train, y_train)
    log_epochs = epochs / 100
    loss_hist = [0] * epochs
    accuracy_hist = [0] * epochs

    for epoch in range(epochs):
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = lossFunction(pred, y_batch.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) ==
                          y_batch).float()
            accuracy_hist[epoch] += is_correct.sum()
            loss_hist[epoch] /= len(train_dl.dataset)
            accuracy_hist[epoch] /= len(train_dl.dataset)
        if epoch % log_epochs == 0:
            print(f"Epoch {epoch} Loss {loss_hist[epoch]:.10f}")
    print(f"Epoch {1000} Loss {loss_hist[1000 - 1]:.10f}")
    return model, loss_hist, accuracy_hist


def plot_loss_accuracy(loss_hist, accuracy_hist, name: str):
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


def plot_decision_boundary(X_train, Y_train, X_cv, Y_cv, model, name: str):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train,
               marker=".", cmap=cmap_dataset2)
    ax.set_xticks(range(-3, 4))
    ax.set_yticks(range(-3, 4))
    ax.set_title('Training data', size=15)
    ax.set_xlabel('X0', size=15)
    ax.set_ylabel('X1', size=15)
    xx, yy = np.meshgrid(np.arange(-3, 3, 0.01),
                         np.arange(-3, 3, 0.01))
    Z = model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())
    Z = torch.argmax(Z, dim=1).reshape(xx.shape)
    ax.contour(xx, yy, Z, alpha=1, colors=['darkgreen'], linewidths=[2])
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(X_cv[:, 0], X_cv[:, 1], c=Y_cv, marker="<", cmap=cmap_dataset1)
    ax.set_xticks(range(-3, 4))
    ax.set_yticks(range(-3, 4))
    ax.set_title('Cross validation data', size=15)
    ax.set_xlabel('X0', size=15)
    ax.set_ylabel('X1', size=15)
    xx, yy = np.meshgrid(np.arange(-3, 3, 0.01),
                         np.arange(-3, 3, 0.01))
    Z = model(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())
    Z = torch.argmax(Z, dim=1).reshape(xx.shape)
    ax.contour(xx, yy, Z, alpha=1, colors=['darkgreen'], linewidths=[2])
    plt.tight_layout()
    plt.savefig(f'{plot_folder}/{name}.png', dpi=300)
    plt.clf()


def evaluate_model(X_train, y_train, X_cv, y_cv, X_test, y_test, model):
    X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
    X_train_norm = torch.from_numpy(X_train_norm).float()
    y_train = torch.from_numpy(y_train)
    pred = model(X_train_norm)
    train_acc = (torch.argmax(pred, dim=1) == y_train).float().mean()
    print(f"Train accuracy: {train_acc:.4f}")
    X_cv_norm = (X_cv - np.mean(X_cv)) / np.std(X_cv)
    X_cv_norm = torch.from_numpy(X_cv_norm).float()
    y_cv = torch.from_numpy(y_cv)
    pred = model(X_cv_norm)
    cv_acc = (torch.argmax(pred, dim=1) == y_cv).float().mean()
    print(f"CV accuracy: {cv_acc:.4f}")
    X_test_norm = (X_test - np.mean(X_test)) / np.std(X_test)
    X_test_norm = torch.from_numpy(X_test_norm).float()
    y_test = torch.from_numpy(y_test)
    pred = model(X_test_norm)
    test_acc = (torch.argmax(pred, dim=1) == y_test).float().mean()
    print(f"Test accuracy: {test_acc:.4f}")


def plot_data(X_train, y_train, X_cv, y_cv, centers, radius, name: str):

    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=y_train, marker=".", cmap=cmap_dataset2)
    plt.scatter(X_cv[:, 0], X_cv[:, 1], c=y_cv, marker='<', cmap=cmap_dataset1)
    circles = [plt.Circle(centers[i], radius * 2, color=cmap_dataset2(i), fill=False)
               for i in range(6)]
    for circle in circles:
        plt.gca().add_artist(circle)

    plt.savefig(f'{plot_folder}/{name}.png', dpi=300)


def generate_data_driver(commandLine: commandline.CommandLine):
    plt.clf()
    X, y, center = generate_data()
    X_train, X_cv, X_test, y_train, y_cv, y_test = train_split(X, y)

    if not os.path.exists(f'{plot_folder}/dataset.png'):
        plot_data(X_train, y_train, X_cv, y_cv, center, 0.4, 'dataset')
    elif commandLine.plot or commandLine.all:
        plot_data(X_train, y_train, X_cv, y_cv, center, 0.4, 'dataset')
    return X_train, X_cv, X_test, y_train, y_cv, y_test


def simple_model_driver(X_train, y_train, X_cv, y_cv, X_test, y_test, commandLine):
    plt.clf()
    if not os.path.exists(f'{plot_folder}/loss_accuracy_simple.png') or commandLine.all or commandLine.simple:
        model, loss_hist, accuracy_hist = simple_model(
            X_train, y_train)
        plot_loss_accuracy(loss_hist, accuracy_hist, 'loss_accuracy_simple')
        evaluate_model(X_train, y_train, X_cv, y_cv, X_test, y_test, model)
        plt.title('Simple model')
        plot_decision_boundary(X_train, y_train, X_cv, y_cv, model,
                               'decision_boundary_simple')


def complex_model_driver(X_train, y_train, X_cv, y_cv, X_test, y_test, commandLine):
    plt.clf()
    if not os.path.exists(f'{plot_folder}/loss_accuracy_complex.png') or commandLine.all or commandLine.complex:
        model, loss_hist, accuracy_hist = complex_model(
            X_train, y_train)
        plot_loss_accuracy(loss_hist, accuracy_hist, 'loss_accuracy_complex')
        evaluate_model(X_train, y_train, X_cv, y_cv, X_test, y_test, model)
        plt.title('Complex model')
        plot_decision_boundary(X_train, y_train, X_cv, y_cv, model,
                               'decision_boundary_complex')


def main():
    commandLine = commandline.CommandLine()
    commandLine.parse(sys.argv[1:])
    X_train, X_cv, X_test, y_train, y_cv, y_test = generate_data_driver(
        commandLine)

    complex_model_driver(X_train, y_train, X_cv, y_cv,
                         X_test, y_test, commandLine)
    simple_model_driver(X_train, y_train, X_cv, y_cv,
                        X_test, y_test, commandLine)


if __name__ == "__main__":
    main()
