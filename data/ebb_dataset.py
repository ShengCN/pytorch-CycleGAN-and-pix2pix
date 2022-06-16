"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
import torchvision.transforms as T
import logging
import random
import h5py
import numpy as np

class EbbDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--hdf5_file', type=str,  help='where is the hdf5_file?')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        # self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.transform = get_transform(opt)

        random.seed(19920208)
        np.random.seed(19920208)

        self.is_training = True
        self.transform   = T.Compose([T.ToTensor()])
        self.fact        = 0.1
        self.update_meta(opt.hdf5_file)


    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path = 'temp'    # needs to be a string

        if not self.is_training:
            idx += self.train_size

        x = self.ds_x[self.ind[idx]]
        y = self.ds_y[self.ind[idx]]

        x    = self.transform(x)
        y    = self.transform(y)

        return {'A': x, 'B': y}
        # return {'data_A':x, 'data_B': y}

        # return {'data_A': data_A, 'data_B': data_B, 'path': path}


    def __len__(self):
        if self.is_training:
            return self.train_size
        else:
            return self.valid_size


    def update_meta(self, hdf5_file):
        f = h5py.File(hdf5_file, 'r')

        self.ds_x  = f['train_x_imgs_256']
        self.ds_y  = f['train_y_imgs_256']
        if len(self.ds_x) != len(self.ds_y):
            err = 'x and y not equal. {} v.s. {}'.format(len(self.ds_x), len(self.ds_y))
            logging.error(err)
            raise ValueError(err)

        self.ind = [i for i in range(len(self.ds_x))]

        logging.info("There are {} files".format(len(self.ds_x)))

        random.shuffle(self.ind)

        total_n = int(len(self.ind) * self.fact)
        print('Total dataset size: ', total_n)

        self.train_size = total_n // 10 * 9
        self.valid_size = total_n - self.train_size
