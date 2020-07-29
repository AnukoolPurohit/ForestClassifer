"""
    This file contains Logger Class which provides functionality for storing
    and ploting metadata (training loss, validation loss, and learning rates)
    for the training process done in NNTrainer class.
"""
import matplotlib.pyplot as plt
from Planet.callbacks import Callback

class Logger(Callback):
    """ Logger object logs changes in learning rate, validation_loss, and
        training loss. Which can later be plotted as a graph.

    Attributes
    ----------
    data : dict
        Dictionary containing training loss, validation loss, and
        learning rate.

    """

    def __init__(self, trainer):
        self.trainer = trainer
        
        self.trainingLoss = []
        self.validationLoss = []
        self.learningRates = []

        self.epochTrainLoss = []
        self.epochValidLoss = []
        return
    
    def log_lr(self):
        self.learningRates.append(self.trainer.opt.param_groups[-1]['lr'])

    def log_loss(self):
        if self.trainer.in_train:
            self.epochTrainLoss.append(self.trainer.loss.detach().item())
        else:
            self.epochValidLoss.append(self.trainer.loss.detach().item())

    def epoch_reset(self):
        train_loss = round(sum(self.epochTrainLoss)/len(self.epochTrainLoss), 6)
        valid_loss = round(sum(self.epochValidLoss)/len(self.epochValidLoss), 6)
        self.trainer.innerbar.write_data([train_loss, valid_loss])
        
        self.trainingLoss.extend(self.epochTrainLoss)
        self.validationLoss.extend(self.epochValidLoss)

        self.epochTrainLoss, self.epochValidLoss = [],[]
        return

    def plot_loss(self):
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(self.trainingLoss, 'b', label='Training Loss')
        ax[0].legend(loc='best')
        ax[1].plot(self.validationLoss, 'y', label='Validation Loss')
        ax[1].legend(loc='best')
        fig.suptitle('Training Summary')
        plt.show()
    
    def plot_lr(self):
        plt.plot(self.learningRates)
        plt.show()
