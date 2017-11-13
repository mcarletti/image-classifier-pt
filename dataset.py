from __future__ import print_function
import torch
import scipy.io as sio
from PIL import Image
import numpy as np


#%% ---------------------------------------------------------------------------


class DATASET(torch.utils.data.Dataset):
    '''
    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
    
    '''

    def __init__(self, train=True, transform=None):
        if train:
            self.matfile = './data/bin_data/train.mat'
        else:
            self.matfile = './data/bin_data/test.mat'
        self.transform = transform

        data = sio.loadmat(self.matfile)
        self.images = np.squeeze(data['images'])
        self.values = np.squeeze(data['values'])
        self.labels = np.squeeze(data['labels'])

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index.

        Returns:
            tuple: (image, target, label).
        '''
        image = self.images[index]
        target = self.values[index]

        image = Image.fromarray(image.squeeze())
        if self.transform is not None:
            image = self.transform(image)

        return image, target, str(self.labels[target])

    def __len__(self):
        return len(self.images)
