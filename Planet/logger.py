"""
    This file contains Logger Class which provides functionality for storing
    and ploting metadata (training loss, validation loss, and learning rates)
    for the training process done in NNTrainer class.
"""
import matplotlib.pyplot as plt


class Logger():
    """ Logger object logs changes in learning rate, validation_loss, and
        training loss. Which can later be plotted as a graph.

    Attributes
    ----------
    data : dict
        Dictionary containing training loss, validation loss, and
        learning rate.

    """

    def __init__(self):
        self.data = {
            'training-loss': [],
            'validation-loss': [],
            'learning-rate': []
        }
        return

    def log_data(self, data):
        for key in self.data.keys():
            self.data[key].extend(data[key])
        return

    def los_lists(self, listdata):
        data = {
            'training-loss': listdata[0],
            'validation-loss': listdata[1],
            'learning-rate': listdata[2]
        }
        self.log_data(data)
        return

    def plot_loss(self):
        training_loss = self.data['training-loss']
        validation_loss = self.data['validation-loss']
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(training_loss, 'b', label='Training Loss')
        ax[0].legend(loc='best')
        ax[1].plot(validation_loss, 'y', label='Validation Loss')
        ax[1].legend(loc='best')
        fig.suptitle('Training Summary')
        plt.show()
