"""
    This is the main file for creating the submission csv for the kaggle
    competition. This goes through both test data and additional test data
    folder combining the predictions from the model for both these set of
    images into a single csv.
"""
from Planet.data import PlanetDataCollection
from Planet.utils import get_transforms
from Planet.train import NNTrainer
from torchvision import models
from pathlib import Path
import pandas as pd
import sys


def print_info(info):
    """ Printing information about what stage of predictions are getting
        generated.
    """
    str_format = "{:^75}"
    print('-'*75)
    print(str_format.format(info))
    print('-'*75)
    return


def get_df(preds, data, thresh=0.2):
    """ Converts predictions into a pandas dataframe.
    """
    labeled_preds = [' '.join([data.train_ds.categories[i]
                               for i, p in enumerate(pred)
                               if p > thresh]) for pred in preds]
    fnames = data.test_ds.x.filenames
    df = pd.DataFrame({'image_name': fnames, 'tags': labeled_preds},
                      columns=['image_name', 'tags'])
    df['extra'] = df['image_name'].str.split('_', n=1, expand=True)[1]
    df['extra'] = df['extra'].astype(float)
    df.sort_values(by='extra', inplace=True)
    df.drop(columns='extra', inplace=True)
    return df


def main(path, model_name):
    """ Main function that generates prediction all all test data and then
        stores it as a .csv file.

    Parameters
    ----------
    path : str
        Loaction on disk where the Planet datset is stored.
    model_name : str
        The model that you will be useing for these predictions. Trained models
        are stored in Models folder.
    """
    # Converting path into Posix path
    path = Path(path)
    # Generating torchvision transforms
    tfms = get_transforms()
    # PlanetDataCollection object contains all the data needed
    data = PlanetDataCollection.from_csv(path, 'train_v2.csv', 'train-jpg',
                                         tfms, name_col='image_name',
                                         label_col='tags', delimter=' ', bs=64,
                                         test_folder='test-jpg')
    # Same Resnet as the one with which trained model was built.
    arch = models.resnet50
    # Creating a NNTrainer object
    trainer = NNTrainer(data, arch)
    """
        Important Step: Here we load the model we have already trained into
        this NNTrainer object.
    """
    trainer.load(model_name)

    # Generating prediction for test-jpg folder
    print_info("Predicting for test-jpg folder")
    preds1 = trainer.predict_test()

    # Converting predictions into a dataframe
    df1 = get_df(preds1, data)
    """
        New PlanetDataCollection object with test set taken from
        test-jpg-additional folder so that we can take prediction for images
        in this folder as well.
    """
    data2 = PlanetDataCollection.from_csv(path, 'train_v2.csv', 'train-jpg',
                                          tfms, name_col='image_name',
                                          label_col='tags',
                                          delimter=' ', bs=64,
                                          test_folder='test-jpg-additional')
    """
        Replacing data in NNTrainer object so that predictions can drawn from
        test set created from test-jpg-additional folder.
    """
    trainer.data = data2

    # Generating predictipns for test-jpg-additional folder
    print_info("Predicting for test-jpg-additional folder")
    preds2 = trainer.predict_test()

    # Converting predictions into dataframe.
    df2 = get_df(preds2, data2)

    # Combining two dataframes for diffrent test sets into one.
    df_sub = pd.concat([df1, df2])

    # Saving the dataset as .csv file.
    print_info("Creating submission.csv")
    df_sub.to_csv('submission.csv', index=False)
    return


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error No Path and Model name given")
    elif len(sys.argv) == 2:
        print("No Model name given")
    else:
        path = sys.argv[1]
        model_name = sys.argv[2]
        main(path, model_name)
