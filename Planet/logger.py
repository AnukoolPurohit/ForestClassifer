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
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(training_loss, 'b', label='Training Loss')
        ax1.legend(loc='best')
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(validation_loss, 'y', label='Validation Loss')
        plt.legend(loc='best')
        plt.show()
