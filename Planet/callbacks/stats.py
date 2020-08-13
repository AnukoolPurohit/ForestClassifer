import time
import torch
from functools import partial
from Planet.callbacks.callback import Callback
from Planet.utils.basic import listify

class AvgStats():
    def __init__(self, metrics, in_train):
        self.metrics, self.in_train = listify(metrics), in_train
    
    def reset(self):
        self.tot_loss, self.count = 0., 0
        self.tot_mets = [0.] * len(self.metrics)
    
    @property
    def all_stats(self):
        return [self.tot_loss.item()] + self.tot_mets
    
    @property
    def avg_stats(self):
        return [o/self.count for o in self.all_stats]
    
    def __repr__(self):
        if not self.count:
            return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"
    
    def accumlate(self, trainer):
        bn = trainer.xb.shape[0]
        self.tot_loss += trainer.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(trainer.pred, trainer.yb) * bn


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats = AvgStats(metrics, True), AvgStats(metrics, False)
        return
    
    def begin_fit(self):
        met_names = ['loss']
        for m in self.train_stats.metrics:
            if isinstance(m, partial):
                met_names.append(m.func.__name__)
            else:
                met_names.append(m.__name__)
        self.names = ['epoch'] + [f'train_{n}' for n in met_names] + [
            f'valid_{n}' for n in met_names] + ['time [min:sec]']
        self.pattern = " ".join(["{:^15}"]*(len(self.names)))

    
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()
        return
    
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumlate(self.trainer)
        return
    
    def after_epoch(self):
        if self.epoch == 0:
            self.innerbar.write(self.pattern.format(*self.names))
        stats = [str(self.epoch)]
        for stat in [self.train_stats, self.valid_stats]:
            stats += [f'{v:.6f}' for v in stat.avg_stats]
        stats += [time.strftime('%M:%S', time.gmtime(time.time() -self.start_time))]
        self.innerbar.write(self.pattern.format(*stats))