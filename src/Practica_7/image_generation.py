import matplotlib.pyplot as plt

import numpy as np

# Custom DPI for the different plots, 300 is very high quality but makes PDF files not able to be submitted to the course assignment system, for that case use 100
customDPI = 300

# Folder where to save the plots

plot_folder = 'memoria/plots/'


def plot_confusion_matrix(y: np.ndarray, p: np.ndarray, tags: list[str], filename: str) -> None:
    """Plots the confusion matrix for a given prediction.

    Args:
        y (np.ndarray): expected values
        p (np.ndarray): predicted values
        tags (list[str]): list of tags
        filename (str): file to store the plot
    """
    fig, ax = plt.subplots()
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_xticks(np.arange(0, tags))
    ax.set_yticks(np.arange(0, tags))
    ax.set_ylabel("True")
    cm = np.zeros((tags, tags))
    for i in range(len(y)):
        cm[y[i] - 1][p[i] - 1] += 1
    cax = ax.matshow(cm, cmap='Reds')
    ax.set_xticks(np.arange(0, tags))
    ax.set_yticks(np.arange(0, tags))
    ax.set_yticks(np.arange(0.5, tags + 0.5), minor='True')
    ax.set_xticks(np.arange(0.5, tags + 0.5), minor='True')
    plt.grid(which='minor', color='lightgrey', linestyle='-', linewidth=0.5)
    fig.colorbar(cax)
    for (i, j), z in np.ndenumerate(cm):
        # Adjust text color based on intensity of background
        text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        ax.text(j, i, '{:0.1f}'.format(z), ha='center',
                va='center', fontsize=8, color=text_color)
    plt.savefig(plot_folder + filename, dpi=customDPI)


def plot_scatter(x: np.ndarray, y: np.ndarray, _class: np.ndarray, class_marker: list[str] = ['o', 'x'], filename: str = 'scatter_plot.png') -> None:
    """Plots a scatter plot for the given data.

    Args:
        x (np.ndarray): x values
        y (np.ndarray): y values
        _class (np.ndarray): class values
        class_marker (list[str], optional): list of markers for each class. Defaults to ['o', 'x'].
        filename (str, optional): file to store the plot. Defaults to 'scatter_plot.png'.
    """
    fig, ax = plt.subplots()
    for i in range(len(class_marker)):
        ax.scatter(x[_class == i + 1], y[_class == i + 1],
                   marker=class_marker[i])
    plt.savefig(plot_folder + filename, dpi=customDPI)
