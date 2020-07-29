"""
     This file contains NNTrainer class that does the main task of bringing
     together data from PlanetDataCollection class and classfier from
     PlanetClassifer class and trains the model.
"""
from torch import nn
from torch import optim
import os
import pickle
import torch
from Planet.progress import InnerBar, OuterBar
from Planet.model import PlanetClassifer
from Planet.scheduler import ParamScheduler
from Planet.logger import Logger
from torch import tensor
from tqdm.auto import tqdm

class NNTrainer():
    """ Class for combing all elements involved in training a Nueral Network.
        Here we combine data with model and data logging, to train and monitor
        the Nueral Network.

    Parameters
    ----------
    data : Object of type PlanetDataCollection
        Contains all data needed including training set, validation set and
        optionally test set.
    arch : Pytorch nn.Module (Pretrained Resnet)
        Pretrained Resnet which will be used as the base by PlanetClassifer
        Constructer.

    Attributes
    ----------
    model : Object of type PlanetClassifer
        Model for Clasfier that will be trained
    loss_func : Pytorch nn.Module
        Loss Function
    log : Object of type Logger
        Log for recording meta data related to training, ie validation loss,
        training loss, and learning rate.
    data: Object of type PlanetDataCollection
        Contains training set, validation set, and optionally Test Set.

    """

    def __init__(self, data, arch, sched_func, optimizer):
        self.data = data
        self.model = PlanetClassifer(arch, output_sz=data.c)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.log = Logger(self)
        self.innerbar = InnerBar(self, leave=False)
        self.outerbar = OuterBar(self, leave=False)
        self.sched = ParamScheduler('lr', sched_func, self)
        self.optimizer = optimizer
        return

    def freeze(self):
        """ Freezing all layers of the model except the final few fully
            connected layer.
        """
        self.model.freeze()
        return

    def unfreeze(self):
        """ Unfreezing all layers of the model.
        """

        self.model.unfreeze()
        return

    def one_batch(self, xb, yb):
        self.xb, self.yb = xb, yb

        if self.in_train:
            self.sched.set_params()
        
        self.pred = self.model(self.xb)
        self.loss = self.loss_func(self.pred, self.yb)
        self.log.log_loss()
        if self.in_train:
            self.loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return
    
    def all_batches(self, dl):
        self.tl = 0.
        self.iters = len(dl)
        for xb, yb in self.innerbar(dl):
            self.one_batch(xb, yb)
            if self.in_train:
                self.log.log_lr()
                self.n_epochs += 1./self.iters
                self.n_iter += 1
            self.innerbar.update()
        return
    
    def fit(self, epochs, lr=1e-3):
        self.epochs = epochs
        self.loss = tensor(0.)
        self.n_epochs = 0.
        self.n_iter = 0
        self.opt = self.optimizer(self.model.parameters(), lr)
        for epoch in self.outerbar(range(epochs)):    
            self.epoch = epoch
           
            self.model.train()
            self.in_train = True
            # Training Loop
            self.all_batches(self.data.train_dl)

            self.model.eval()
            # Validation Loop
            self.in_train = False
            with torch.no_grad():
                self.all_batches(self.data.valid_dl)
            self.log.epoch_reset()
        return
    
    def save(self, filename):
        """ Saving the trained model in Models folder.
        Parameters
        ----------
        filename: str
            Name of file.
        """
        if not os.path.exists('Models'):
            os.mkdir('Models')
        torch.save(self.model.state_dict(), 'Models/'+filename)
        return
    
    def load(self, filename):
        """ Loading a trained model from a file in Models folder

        Parameters
        ----------
        filename: str
            Name of pickle file.
        """

        self.model.load_state_dict(torch.load('Models/'+filename))
        return

    def predict_batch(self, batch):
        """ Predicting outputs for a single batch. Used for Test set.
        """
        self.freeze()
        self.model.eval()
        res = self.model(batch)
        return res

    def predict_test(self):
        """ Predicting ouputs for the whole Test set.
        """
        res = []
        for x, _ in tqdm(self.data.test_dl):
            res.append(self.predict_batch(x))
        res = torch.cat(res)
        return nn.Sigmoid()(res)
