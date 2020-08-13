"""
     This file contains NNTrainer class that does the main task of bringing
     together data from PlanetDataCollection class and classfier from
     PlanetClassifer class and trains the model.
"""
import os
import pickle
import torch
from Planet.callbacks.progress import InnerBar, OuterBar
from Planet.callbacks.scheduler import ParamScheduler
from Planet.callbacks.training import TrainEvalCallback

from Planet.exceptions import CancelBatchException
from Planet.exceptions import CancelEpochException
from Planet.exceptions import CancelTrainException

from Planet.utils.basic import listify
from Planet.model import PlanetClassifer

from torch import nn
from torch import optim
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

    def __init__(self, data, model, loss_func, opt, lr=1e-2,
                     cbs=None, cb_funcs=None):
        self.model, self. data , self.loss_func = model, data, loss_func
        self.opt, self.lr = opt, lr
        
        self.in_train = False

        self.innerbar = InnerBar(self, leave=False)
        self.outerbar = OuterBar(self, leave=False)
        
        self.cbs = []
        self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cb_funcs))
        return
    
    def add_cbs(self, cbs):
        for cb in listify(cbs):
            self.add_cb(cb)
        return
    
    def add_cb(self, cb):
        cb.set_trainer(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)
        return
    
    def remove_cbs(self, cbs):
        for cb in cbs:
            self.cbs.remove(cb)
        return

    def one_batch(self, xb, yb):
        try:
            self.xb, self.yb = xb, yb
            self('begin_batch')
            
            self.pred = self.model(self.xb)
            self('after_pred')
            
            self.loss = self.loss_func(self.pred, self.yb)
            self('after_loss')

            if not self.in_train: return
            
            
            self.loss.backward()
            self('after_backward')
            
            self.opt.step()
            self('after_step')
            
            self.opt.zero_grad()

        except CancelBatchException:
            self('after_cancel_batch')
        
        finally:
            self('after_batch')
        return
    
    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb, yb in self.innerbar(dl):
                self.one_batch(xb, yb)
                self.innerbar.update()

        except CancelEpochException:
            self('after_cancel_epoch')
        return
    
    def fit(self, epochs, lr=1e-3):
        self.epochs, self.loss = epochs, tensor(0.)
        
        try:
            for cb in self.cbs:
                cb.set_trainer(self)
            self('begin_fit')
            for epoch in self.outerbar(range(epochs)):    
                self.epoch = epoch
                
                # Training Loop
                if not self('begin_epoch'):
                    self.all_batches(self.data.train_dl)
                # Validation Loop
                with torch.no_grad():
                    if not self('begin_validate'):
                        self.all_batches(self.data.valid_dl)
                self('after_epoch')
        except CancelTrainException:
            self('after_cancel_train')
        finally:
            self('after_fit')
        return
    
    def __call__(self, cb_name):
        res = True
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) and res
        return res

    def predict_batch(self, batch):
        """ Predicting outputs for a single batch. Used for Test set.
        """
        self.model.cuda()
        self.model.freeze()
        self.model.eval()
        res = self.model(batch)
        return res

    def predict_test(self):
        """ Predicting ouputs for the whole Test set.
        """
        res = []
        for x, _ in tqdm(self.data.test_dl):
            x = x.cuda()
            res.append(self.predict_batch(x))
        res = torch.cat(res)
        return nn.Sigmoid()(res)