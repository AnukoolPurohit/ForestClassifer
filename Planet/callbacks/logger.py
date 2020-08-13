"""
    This file contains Logger Class which provides functionality for storing
    and ploting metadata (training loss, validation loss, and learning rates)
    for the training process done in NNTrainer class.
"""
import matplotlib.pyplot as plt
from Planet.callbacks.callback import Callback
from Planet.utils.basic import listify


class Logger(Callback):
    """ Logger object logs changes in learning rate, validation_loss, and
        training loss. Which can later be plotted as a graph.

    Attributes
    ----------
    data : dict
        Dictionary containing training loss, validation loss, and
        learning rate.

    """

    def begin_fit(self):        
        self.train_losses = []
        self.valid_losses = []
        self.lrs =[[] for _ in self.opt.param_groups]
        return
    
    def after_batch(self):
        if not self.in_train:
            self.valid_losses.append(self.loss.detach().cpu())
            return
        for pg, lr in zip(self.opt.param_groups, self.lrs):
            lr.append(pg['lr'])
        self.train_losses.append(self.loss.detach().cpu())

    def plot(self, skip_last=0, pgid=-1):
        losses = [loss.item() for loss in self.train_losses]
        lrs = self.lrs[pgid]
        n = len(losses) - skip_last
        plt.xscale('log')
        plt.plot(lrs[:n], losses[:n])

    def plot_loss(self, skip_last=[0,0]):
        assert len(skip_last) == 2
        
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        
        ax[0].plot(self.train_losses[:len(self.train_losses)-skip_last[0]],
                         'b', label='Training Loss')
        ax[0].legend(loc='best')

        ax[1].plot(self.valid_losses[:len(self.valid_losses)-skip_last[1]],
                         'y', label='Validation Loss')
        ax[1].legend(loc='best')
        
        fig.suptitle('Training Summary')
    
    def plot_lr(self, pgid=-1):
        plt.plot(self.lrs[pgid])
        plt.xlabel('iterations')
        plt.ylabel('learning rate')
        plt.title('learning rate schedule')
