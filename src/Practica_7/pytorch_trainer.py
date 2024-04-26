import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as sio
import time
import torch

# Select cuda device if available to speed up training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print(f'Using GPU {torch.cuda.get_device_name()}')
else:
    print('Using CPU')


def train_data(x: np.ndarray, y: np.ndarray) -> torch.utils.data.DataLoader:
    """ Create a DataLoader object from the input data 
    Args:
        x (np.ndarray): Input data
        y (np.ndarray): Target data
    """
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.tensor(x, dtype=torch.float).to(device), torch.tensor(y).to(device)), batch_size=2, shuffle=True)


def train_model(model: nn.Sequential, train_dl: torch.utils.data.DataLoader, criterion: nn.CrossEntropyLoss, optimizer: optim.Adam, epochs: int) -> nn.Sequential:
    """ Train the model with the given data
    Args:
        model (nn.Sequential): Model to train
        train_dl (torch.utils.data.Dataloader): DataLoader object with the training data
        criterion (nn.CrossEntropyLoss): Loss function
        optimizer (optim.Adam): Optimizer
        epochs (int): Number of epochs to train the model
    Returns:
        nn.Sequential: Trained model
    """
    for epoch in range(epochs):
        model.train()
        for x, y in train_dl:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
    return model


def ComplexModel(input_size: int) -> nn.Sequential:
    """Creates a Sequential model with 3 layers

    Args:
        input_size (int): input size of the model

    Returns:
        nn.Sequential: base model
    """
    return nn.Sequential(
        nn.Linear(input_size, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
        nn.ReLU(),
        nn.Linear(10, 2),
        nn.Sigmoid()
    ).to(device)


def pred_check(pred: torch.Tensor, y: np.ndarray) -> float:
    """Gives the accuracy of the model in percentage

    Args:
        pred (torch.Tensor): predictions made by  the model
        y (np.ndarray): target data

    Returns:
        float: predict percentage
    """
    return (pred.argmax(dim=1) == torch.tensor(y).to(device)).sum().item() / len(y)


def trainer(X: np.ndarray, y: np.ndarray) -> None:
    """Trains the model with the given data
    Args: 
        X (np.ndarray): Input data
        y (np.ndarray): Target data
    """

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # y = np.array([[0, 1] if i == 1 else [1, 0] for i in y])

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=22)
    x_cv, x_test, y_cv, y_test = train_test_split(
        x_test, y_test, test_size=0.5, random_state=22)
    lambdas = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.3])
    learning_rates = np.array([0.01, 0.1, 0.5, 1])
    best_score = 0
    best_params = (0, 0)

    for lambda_ in lambdas:
        for learning_rate in learning_rates:
            criterion = nn.CrossEntropyLoss().to(device)
            print(f'Lambda: {lambda_}, Learning rate: {learning_rate}')
            model = ComplexModel(x_train.shape[1])
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=lambda_)
            train_dl = train_data(x_train, y_train)
            model = train_model(
                model, train_dl, criterion, optimizer, 20)
            pred = model(torch.tensor(x_cv, dtype=torch.float).to(device))

            score = pred_check(pred, y_cv)
            print(score)
            if score > best_score:
                best_score = score
                best_params = (lambda_, learning_rate)

    print(f'Best score: {best_score}')
    print(f'Best params: {best_params}')

    start = time.time()
    criterion = nn.CrossEntropyLoss().to(device)
    model = ComplexModel(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=best_params[1],
                           weight_decay=best_params[0])
    train_dl = train_data(x_train, y_train)
    model = train_model(model, train_dl, criterion, optimizer, 20)
    end = time.time()

    test_score = pred_check(
        model(torch.tensor(x_test, dtype=torch.float).to(device)), y_test)
    cv_score = pred_check(
        model(torch.tensor(x_cv, dtype=torch.float).to(device)), y_cv)
    train_score = pred_check(
        model(torch.tensor(x_train, dtype=torch.float).to(device)), y_train)

    print(f'Test score: {test_score}')
    print(f'CV score: {cv_score}')
    print(f'Train score: {train_score}')
    print(f'Time: {end-start}')

    sio.savemat('res/pytorch.mat', {
        'test_score': test_score,
        'cv_score': cv_score,
        'train_score': train_score,
        'best_params': best_params,
        'time': end-start
    })
