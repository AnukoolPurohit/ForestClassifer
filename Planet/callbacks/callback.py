import re
from Planet.utils.callbacks import camel2snake

class Callback():
    _order = 0
    def set_trainer(self, trainer):
        self.trainer = trainer
    
    def __getattr__(self, k):
        return getattr(self.trainer, k)
    
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')
    
    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f():
            return True
        return False