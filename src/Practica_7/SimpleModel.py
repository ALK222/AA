import torch.nn as nn


def SimpleModel() -> nn.Sequential:
    """Model with 2 layers neural network to classify the data:
        - Dense layer with 6 units and relu activation function
        - Dense layer with 6 units and linear activation function
        Returns:
        nn.Sequential: model
    """
    return nn.Sequential(
        nn.Linear(2, 6),
        nn.ReLU(),
        nn.Linear(6, 6)
    )
