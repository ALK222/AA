import torch.nn as nn


def ComplexModel() -> nn.Sequential:
    """Model with 3 layers neural network to classify the data:
        - Dense layer with 120 units and relu activation function
        - Dense layer with 40 units and relu activation function
        - Dense layer with 6 units and linear activation function

    Returns:
        nn.Sequential: model
    """
    return nn.Sequential(
        nn.Linear(2, 120),
        nn.ReLU(),
        nn.Linear(120, 40),
        nn.ReLU(),
        nn.Linear(40, 6)
    )
