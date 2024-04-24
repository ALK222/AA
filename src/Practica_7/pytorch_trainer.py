import torch.nn as nn
import torch.optim as optim
import numpy as np
import evaluation
from sklearn.model_selection import train_test_split
import scipy.io as sio
import time
import torch
import concurrent.futures

torch.set_default_device(
    "cuda:0" if torch.cuda.is_available() else "cpu")


def ComplexModel(input_size):
    return nn.Sequential(
        nn.Linear(input_size, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )


def trainer(X: np.ndarray, y: np.ndarray):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=22)
    x_cv, x_test, y_cv, y_test = train_test_split(
        x_test, y_test, test_size=0.5, random_state=22)
    lambdas = np.array([0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3])
    learning_rates = np.array([0.001, 0.01, 0.1, 0.5, 1])
    criterion = nn.BCELoss()
    best_score = 0
    best_params = (0, 0)

    for lambda_ in lambdas:
        for learning_rate in learning_rates:
            model = ComplexModel(x_train.shape[1]).to(device)
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=lambda_)
            train_dl = evaluation.train_data(x_train, y_train)
            model, _, _ = evaluation.train_model(
                model, train_dl, criterion, optimizer, 1000)
            score = evaluation.evaluate_model(
                x_train, y_train, x_cv, y_cv, x_test, y_test, model)['cv']
            if score > best_score:
                best_score = score
                best_params = (lambda_, learning_rate)

    print(f'Best score: {best_score}')
    print(f'Best params: {best_params}')

    start = time.time()
    model = ComplexModel(X)
    optimizer = optim.Adam(model.parameters(), lr=best_params[1],
                           weight_decay=best_params[0])
    train_dl = evaluation.train_data(x_train, y_train)
    model = evaluation.train_model(model, train_dl, criterion, optimizer, 1000)
    end = time.time()

    scores = evaluation.evaluate_model(
        x_train, y_train, x_cv, y_cv, x_test, y_test, model)

    test_score = scores['test']
    cv_score = scores['cv']
    train_score = scores['train']

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
