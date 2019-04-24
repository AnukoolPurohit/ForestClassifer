"""
    This is the main file for creating and training a CNN Classifier which will
    be trained on Amazon from space dataset by Planet Labs.
"""
from Planet.data import PlanetDataCollection
from Planet.utils import get_transforms
from Planet.train import NNTrainer
from torchvision import models
from pathlib import Path
import sys


def print_phase_info(phase_no, epochs, lr):
    """ Prints meta info about how the networks is being currently trained.
    """
    print('-'*75)
    row_format = "{:>25}" * 3
    print(row_format.format(f"Phase: {phase_no}", f"Epochs: {epochs} |",
                            f"Base Learning Rate: {lr}"))
    print('-'*75)
    return


def main(path):
    """ This function contains the main logic for creating and training the CNN
        Classifier.
    Parameters
    ----------
    path : str
        path to where the dataset is stored on the disk.
    """
    # Converting the path to Posix Path
    path = Path(path)
    # Generating required torchvision transforms we need to apply for
    # preprocessing
    tfms = get_transforms()
    """
    Creating Dataset Collection that contains all three sets, training,
    Validation, and Testing. All Images are retrived directly from the Disk
    and torchvision transforms are applied on these images on the fly as images
    are retrived from the disk which saves RAM.
    """
    data = PlanetDataCollection.from_csv(path, 'train_v2.csv', 'train-jpg',
                                         tfms, name_col='image_name',
                                         label_col='tags', delimter=' ', bs=64,
                                         test_folder='test-jpg')

    # A pretrained Resnet is used as the base for this classifier.
    arch = models.resnet50
    """
    NNtrainer class combines all elements required for training a classfier
    into one object. The class takes as an input a pretrained Resnet and
    A PlanetDataCollection object that has all the data required in correct
    format.
    """
    trainer = NNTrainer(data, arch)

    # Training when with only first few layers unforzen
    print_phase_info(1, 2, 1e-03)
    # Learning Rate is 1e-03 for 2 epochs.
    trainer.train(1e-03, 2)

    print_phase_info(2, 2, 1e-05)
    # Learning Rate is 1e-05 for 2 epochs.
    trainer.train(1e-05, 2)

    print('')
    print('-'*75)
    print('{:^75}'.format('UNFREEZING THE NETWORK'))
    # Unfreezing the Whole Network for finetuning.
    trainer.unfreeze()
    print('-'*75)
    print('')

    print_phase_info(3, 4, 1e-05)
    # Learning Rate is 1e-05 for 4 epochs.
    trainer.train(1e-05, 4)
    # Learning Rate is 1e-06 for 2 epochs.
    print_phase_info(4, 2, 1e-06)
    trainer.train(1e-06, 2)
    # Trained model is saved into a pickle.
    trainer.save('basic_trained')
    # Plotting Training Summary
    trainer.log.plot_loss()
    return


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Error No path given")
    else:
        path = sys.argv[1]
        main(path)
