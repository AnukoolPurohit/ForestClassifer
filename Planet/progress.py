from tqdm.auto import tqdm

class OuterBar():
    def __init__(self, trainer, leave):
        self.trainer = trainer
        self.leave = leave
        return
    
    def progress_bar(self, itr):
        self.bar = tqdm(itr,leave=self.leave)
        self.bar.set_description(f"Epochs")
        return self.bar
    
    def __call__(self, itr):
        return self.progress_bar(itr)

class InnerBar():
    def __init__(self, trainer, leave=True):
        self.trainer = trainer
        self.leave = leave
        return
    
    def progress_bar(self, itr):
        self.bar = tqdm(itr, leave=self.leave)
        stage = "Training " if self.trainer.in_train else "Validation"
        self.bar.set_description(f"Epoch-{self.trainer.epoch}: {stage}")
        loss = self.trainer.loss.detach().item()
        self.bar.set_postfix({f"Loss":"{:.2f}".format(loss)})
        return self.bar
    
    def update(self):
        loss = self.trainer.loss.detach().item()
        self.bar.set_postfix({f"Loss":"{:.2f}".format(loss)})
        self.bar.display(f"{self.trainer.loss.item()}")
        self.bar.update()
        return
    
    def write_data(self, data):
        format_string = "{:^20} {:^20} {:^20}"
        if self.trainer.epoch == 0:
            self.write(format_string.format("Epoch",
                         "Training Loss", "Validation Loss"))
        self.write(format_string.format(self.trainer.epoch,
                 data[0], data[1]))
        return

    def write(self, msg):
        self.bar.write(f"{msg}")
        self.bar.update()
        return

    def __call__(self, itr):
        return self.progress_bar(itr)