"""
    This file contains miscellaneous functions that are required for this
    classifier to work.
"""
from typing import Iterable
from torchvision import transforms
import pandas as pd


def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]


def get_df(preds, data, thresh=0.2):
    """ Convert model predictions into Pandas Dataframe.
    Returns
    -------
    pandas.dataframe
         Containing results from model predictions.

    """
    labeled_preds = [' '.join([data.train_ds.categories[i] for i, p in enumerate(
        pred) if p > thresh]) for pred in preds]
    fnames = data.test_ds.x.filenames
    df = pd.DataFrame({'image_name': fnames, 'tags': labeled_preds}, columns=[
                      'image_name', 'tags'])
    df['extra'] = df['image_name'].str.split('_', n=1, expand=True)[1]
    df['extra'] = df['extra'].astype(float)
    df.sort_values(by='extra', inplace=True)
    df.drop(columns='extra', inplace=True)
    return df


def get_transforms():
    """ Generates torchvision transforms for both validation and training set
        While Random image augmentaion is performed in training set no such
        augmentaion is done for validation set.
    Returns
    -------
    dict
        Containing torchvision transforms for training, validation and
        test set. Note same transforms are used on validation and test set.

    """
    tfms = {
        # Training transforms contains random augmentaions.
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        # Validation transforms contains no augmentaion.
        'valid': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        # Display Transforms to show the effect on training set
        'train_display': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]),
    }
    return tfms
