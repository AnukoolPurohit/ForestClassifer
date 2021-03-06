"""
    This file contains code for creating the Classfier Model. An object of
    PlanetClassifer Class is the model created from a pretrained Resnet.
"""
import os
import torch
from torch import nn


class PlanetClassifer(nn.Module):
    """ Creating Model for Multilabel Classification using pretrained Resnet
        Model. We cleave of the fully connected layer at the end of a
        pretrained Resnet and out new layers with appropriate number of ouputs.

    Parameters
    ----------
    arch : Pytorch Pretrained Model
        The pretrained resnet model which forms the basis for this classifer.
    output_sz : size of output
        Size of output depends on the number of categories.

    Attributes
    ----------
    model : pytorch nn.Module
        Description of attribute `model`.
    """

    def __init__(self, arch, output_sz=2):
        super().__init__()
        model = arch(pretrained=True)
        in_features = model.fc.in_features
        linear = nn.Sequential(nn.Linear(in_features, 256),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(256, output_sz))
        model.fc = linear
        self.model = model
        self.freeze()
        return

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        """ Freezes all layers except the final fully connnected layers that
            are added by this model.
        """
        children = [chlid for chlid in self.model.children()]
        for child in children[0:8]:
            for params in child.parameters():
                params.requires_grad = False
        return

    def unfreeze(self):
        """ Unfreeze all layers
        """
        for params in self.model.parameters():
            params.requires_grad = True
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
        torch.save(self.state_dict(), 'Models/'+filename)
        return
    
    def load(self, filename):
        """ Loading a trained model from a file in Models folder

        Parameters
        ----------
        filename: str
            Name of pickle file.
        """

        self.load_state_dict(torch.load('Models/'+filename))
        return
