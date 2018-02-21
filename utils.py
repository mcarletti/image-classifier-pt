# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable

import scipy.io as sio
import numpy as np

# Conversion ----------------------------------

def numpy_to_torch(array, use_cuda=False):
    tensor = torch.from_numpy(array)
    tensor = Variable(tensor)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor

def torch_to_numpy(tensor):
    if not isinstance(tensor, Variable):
        tensor = Variable(tensor)
    array = tensor.cpu().data.numpy()
    return array

# PyTorch -------------------------------------

def enable_grads(model, enable=True):
    for param in model.parameters():
        model.requires_grad = enable

def normal_init(module, mu=0., std=0.01):
    module.weight.data.normal_(mu, std)
    module.bias.data.fill_(mu)

# Save and load -------------------------------

def savemat(filename, data):
    try:
        sio.savemat(filename, data, do_compression=True)
    except Exception as e:
        raise e

def loadmat(filename):
    try:
        data = sio.loadmat(filename)
        return data
    except Exception as e:
        raise e

# Misc ----------------------------------------

def find_outliers(arr):
    mu = np.mean(arr)
    sd = np.std(arr)
    idx1 = np.where(arr < mu - 2 * sd)[0]
    idx2 = np.where(arr > mu + 2 * sd)[0]
    return np.sort(np.concatenate([idx1, idx2]))