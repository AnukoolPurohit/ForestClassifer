"""
    This file contains all the data storage and preprocessing related elements
    for this Classifier. It contains classes which are involved in retiving,
    preprocessing augmenting, and divding images into diffrent datasets.
    The Ojects are made such that all images are retrived on the fly rather
    than storing them in RAM.
"""
import torch
import re
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class PlanetImages():
    """Object for retreving Images from Disk on the fly.

    Attributes
    ----------
    filenames (iterable):
        Contains filenames for all images.
    path (Posix Path):
        Contains path of the location where the images are stored on disk.

    """

    def __init__(self, filenames, path):
        self.filenames = filenames
        self.path = path
        return

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """ For retriving the image when indexed.
        """
        image = io.imread(self.path / f'{self.filenames[idx]}.jpg')
        image = Image.fromarray(image)
        return image

    def __str__(self):
        return f"Images {[self[i].size for i in range(3)]}..."

    def __repr__(self):
        return f"Images {[self[i].size for i in range(3)]}..."


class PlanetDataset(Dataset):
    """ For creating iterable objects that can generate (x,y) pairs on the fly.
        Inherited from pytorch class Dataset so that it can be used to create
        iterable pytorch Dataloader.

    Attributes
    ----------
    x (Object of Type PlanetImages):
        Contains PlanetImages object contain all images for this dataset.

    y (numpy.ndarray or None):
        Contains all the labels for images genetated from x as one hot encoded
        arrays.

    categories (numpy.ndarray of strings):
        Contains names of categories that dataset is labeled on as strings.

    df (Pandas DataFrame object or None):
        If from_df function is used to create the object then,
        This contains the Dataframe from which the PlanetDataset object is
        created.

    tfms (torchvision transforms compose)
        Contains all transforms that need to be applied to images retrived from
        disk.

    """

    def __init__(self, x, y, categories, df, tfms):
        super().__init__()
        self.x = x
        self.y = y
        self.categories = categories
        self.df = df
        self.tfms = tfms
        return

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """ Retriving Image and appying trasforms when indexed.

        Parameters
        ----------
        idx (int):
            Index

        Returns
        -------
        Tuple of torch tensors containing both the image and one hot encoded
        label.
        """
        x = self.tfms(self.x[idx])
        y = self.y[idx]
        x = x
        y = torch.FloatTensor(y)
        y = y
        return (x, y)

    def __str__(self):
        out = f"""
        x: Images {[self[i][0].shape for i in range(3)]}...
        y: Labels {[
            [self.categories[i] for i, x in enumerate(self.y[j])
             if x == 1]
            for j in range(3)
        ]} ...

        """
        return out

    def __repr__(self):
        out = f"""
        x: Images {[self[i][0].shape for i in range(3)]}...
        y: Labels {[
            [self.categories[i] for i, x in enumerate(self.y[j])
             if x == 1]
            for j in range(3)
        ]} ...

        """
        return out

    @classmethod
    def from_df(cls, df, path, folder, tfms, name_cols,
                label_cols, delimter):
        """ Constructer using pandas DataFrame.
            Used for training and validation datasets.

        Parameters
        ----------
        df (Pandas DataFrame):
            Dataframe containing image filenames and labels.

        path (Posix Path):
            path where the folder and orignal csv are stored.

        folder (str):
            name of the folder.

        tfms (torchvision transform compile):
            Transforms that are to be applied on each image.

        name_cols (str):
            Column containg filenames.

        label_cols (str):
            Column containg labels.

        delimter (str):
            Delimter between categories.

        Returns
        -------
        Object of PlanetDataset type
        """
        items = df[name_cols].tolist()
        x = PlanetImages(items, path / folder)

        mlb = MultiLabelBinarizer()
        tags = df[label_cols].str.split(delimter).tolist()

        y = mlb.fit_transform(tags)
        categories = mlb.classes_

        return cls(x, y, categories, df, tfms)

    @classmethod
    def from_folder(cls, path, folder, tfms):
        """ Constructer using folder.
            Used for test dataset.
        Parameters
        ----------
        path (Posix Path):
            Path to the location on disk where the folder is stored.
        folder (str):
            Folder name.
        tfms (torchvision transforms compile):
            Transforms that are to be applied on each image

        Returns
        -------
        Object of PlanetDataset type
        """
        pat = re.compile('[^.]*')
        items = [re.search(pat, fname)[0] for fname in os.listdir(path/folder)]
        x = PlanetImages(items, path/folder)

        y = [0]*len(x)

        categories = None
        df = None

        return cls(x, y, categories, df, tfms)


class PlanetDataCollection():
    """ For creating objects containg all the data required for training a
        Classifier.
        It creates both the training set and validation set from the same data
        randomly. And has an option to add the test data as well.

    Attributes
    ----------
    train_dl (pytorch DataLoader):
        Training dataloader.

    train_ds (PlanetDataset):
        Training dataset.

    valid_dl (pytorch DataLoader):
        Validation dataloader.

    valid_ds (PlanetDataset):
        Validation dataset.

    tfms (torchvision transform compile):
        Transforms that are to be applied on each image.

    c (int):
        Number of categories.

    test_ds (PlanetDataset):
        Test dataset.

    test_dl (pytorch DataLoader): type
        Test dataloader.
    """

    def __init__(self, train_dl, train_ds, valid_dl, valid_ds,
                 tfms, c, test_ds=None, test_dl=None):
        self.train_ds = train_ds
        self.train_dl = train_dl
        self.valid_ds = valid_ds
        self.valid_dl = valid_dl
        self.test_ds = test_ds
        self.test_dl = test_dl
        self.tfms = tfms
        self.c = c
        return

    def __repr__(self):
        out = f"""
        train_dl: Type torch dataloader
        train_ds: Type PlanetData, len:{len(self.train_ds)}

        x: Images {[self.train_ds[i][0].shape for i in range(3)]}...
        y: Labels {[
            [self.train_ds.categories[i]
             for i,x in enumerate(self.train_ds.y[j])
             if x == 1]
            for j in range(3)
        ]} ...
        valid_dl: Type torch dataloader
        valid_ds: Type PlanetData, len:{len(self.valid_ds)}

        x: Images {[self.valid_ds[i][0].shape for i in range(3)]}...
        y: Labels {[
            [self.valid_ds.categories[i]
             for i,x in enumerate(self.train_ds.y[j])
             if x == 1]
            for j in range(3)
        ]} ...

        """

        return out

    def show(self, nrows=3, figsize=(15, 15)):
        """ Displays a sample from the training set.
        """

        fig, ax = plt.subplots(nrows=nrows, ncols=nrows, figsize=figsize)
        i = 0
        for row in ax:
            for col in row:
                tags = self.train_ds.y[i]
                tags = [i for i, x in enumerate(tags) if x == 1]
                tags = [self.train_ds.categories[i] for i in tags]
                title = ' '.join(tags)
                image = self.train_ds.x[i]
                col.imshow(image)
                col.axis('off')
                col.set_title(title)
                i += 1
        return
    
    def show_tfms(self, nrows=3, figsize=(15, 15)):
        """ Displays a sample to explain the transforms on the training set.
        """

        fig, ax = plt.subplots(nrows=nrows, ncols=nrows, figsize=figsize)
        i = random.randint(0,len(self.train_ds))
        for row in ax:
            for col in row:
                image = self.tfms['train_display'](self.train_ds.x[i])
                col.imshow(image)
                col.axis('off')
        return

    @classmethod
    def from_csv(cls, path, csv_name, folder, tfms, name_col='image_name',
                 label_col='tags', delimter='', bs=64,
                 test_folder=None, pct=0.2):
        """ Constructer for creating the object from a csv file.
            It also controls the batch size and the percentage of data used as
            validation set.

        Parameters
        ----------
        path : Posix Path
            Path to loaction where all the data is on disk.
        csv_name : str
            CSV file containing filenames and labels of training data.
        folder : str
            Name of the folder containg the training images.
        tfms : torchvision transforms compile
            Transforms that are to be applied on each image.
        name_col : str
            Column containg filenames.
        label_col : str
            Column containg Labels.
        delimter : str
            Delimter for breaking categories from label column.
        bs : int
            Batch size.
        test_folder : str or None
            Test Folder name.
        pct : Float
            Percentage of dataset used to create validation set.

        Returns
        -------
        PlanetDataCollection
            Object of class PlanetDataCollection.

        """

        df = pd.read_csv(path / csv_name)
        train_df, valid_df = train_test_split(df, test_size=pct)

        train_ds = PlanetDataset.from_df(train_df, path, folder,
                                         tfms['train'], name_col, label_col,
                                         delimter=delimter)

        valid_ds = PlanetDataset.from_df(valid_df, path, folder,
                                         tfms['valid'], name_col, label_col,
                                         delimter=delimter)

        train_dl = DataLoader(train_ds, batch_size=bs)
        valid_dl = DataLoader(valid_ds, shuffle=False, batch_size=bs)
        if test_folder is None:
            test_dl = None
            test_ds = None
        else:
            test_ds = PlanetDataset.from_folder(
                path, test_folder, tfms['valid'])
            test_dl = DataLoader(test_ds, shuffle=False, batch_size=bs)
        c = len(train_ds.categories)
        return cls(train_dl, train_ds, valid_dl, valid_ds, tfms,
                   c, test_ds, test_dl)
