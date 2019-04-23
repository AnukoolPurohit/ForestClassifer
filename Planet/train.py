from torch import nn
from torch import optim
import os
import pickle
import torch
from tqdm.auto import tqdm
from Planet.model import PlanetClassifer
from Planet.logger import Logger


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

    def __init__(self, data, arch):
        self.data = data
        self.model = PlanetClassifer(arch, output_sz=data.c)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.log = Logger()
        return

    def update(self, x, y, lr):
        """ Executes one step of gradient descent. For images x and labels y at
            learning rate lr.
        """
        self.model.train()
        opt = optim.Adam(self.model.parameters(), lr)
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss.item()

    def evaluate(self, x, y):
        """ Evaluating Model prediction for images x with actual labels y.
        """

        self.model.eval()
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        return loss.item()

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

    def train_epoch(self, lr):
        """ Function Trains the model for one epoch.

        Parameters
        ----------
        lr : float or list
            learning rate for this epoch.

        Returns
        -------
        list
            list containing lists of validation loss, training loss and
            learning rate.

        """
        train_losses = []
        valid_losses = []
        lr_log = []
        train_iter = tqdm(self.data.train_dl, leave=False)
        # Training the model on Training set.
        self.model.train()
        for x, y in train_iter:
            loss = self.update(x, y, lr)
            train_iter.set_postfix({'Loss': loss})
            train_losses.append(loss)
            lr_log.append(lr)
        valid_iter = tqdm(self.data.valid_dl, leave=False)
        # Evaluating the model on Validation set.
        self.model.eval()
        for x, y in valid_iter:
            loss = self.evaluate(x, y)
            valid_iter.set_postfix({'Loss': loss})
            valid_losses.append(loss)
        return [train_losses, valid_losses, lr_log]

    def train(self, lr, epochs=3):
        """ The function used for training the model for n epochs.

        Parameters
        ----------
        lr : float or list
            learning rate or rates.
        epochs : int
            number of epochs.

        """

        training_errors = []
        validation_errors = []
        print_cols = False
        bar = tqdm(range(epochs), leave=False)
        for i in bar:
            res = self.train_epoch(lr)
            self.log.los_lists(res)
            train_error = (sum(res[0])/len(res[0]))
            valid_error = (sum(res[1])/len(res[1]))

            training_errors.append(train_error)
            validation_errors.append(valid_error)

            metalist = ["Training Loss", "Validation Loss"]
            row_format = "{:>25}" * (len(metalist) + 1)
            if print_cols is False:
                print(row_format.format("Epoch", *metalist))
                print_cols = True
            print(row_format.format(i+1, "%.6f" % train_error,
                                    "%.6f" % valid_error))
        return

    def save(self, filename):
        """ Saving the trained model as a pickle in Models folder.

        Parameters
        ----------
        filename: str
            Name of pickle file.
        """

        if not os.path.exists('Models'):
            os.mkdir('Models')
        picklefile = open('Models/'+filename, 'wb')
        pickle.dump(self.model, picklefile)
        picklefile.close()
        return

    def load(self, filename):
        """ Loading a trained model from a pickle file in Models folder

        Parameters
        ----------
        filename: str
            Name of pickle file.
        """

        picklefile = open('Models/'+filename, 'rb')
        self.model = pickle.load(picklefile)
        picklefile.close()
        return

    def predict_batch(self, batch):
        """ Predicting outputs for a single batch. Used for Test set.
        """
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
