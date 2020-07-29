import torch
from Planet.callbacks import Callback


class ParamScheduler(Callback):
    _order = 1
    def __init__(self, pname, sched_func, trainer):
        self.pname = pname
        self.sched_func = sched_func
        self.trainer = trainer
    
    def set_params(self):
        for pg in self.opt.param_groups:
            pg[self.pname] = self.sched_func(self.n_epochs/self.epochs)
    
    def begin_batch(self):
        if self.in_train:
            self.set_params()
