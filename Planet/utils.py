"""
    This file contains miscellaneous functions that are required for this
    classifier to work.
"""
from torchvision import transforms


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
    }
    return tfms
