# -*- coding: utf-8 -*-

from utils import torch_to_numpy
import numpy as np

def accuracy(y_pred, y_targ):
    if not isinstance(y_pred, np.ndarray):
        y_pred = torch_to_numpy(y_pred)
    if not isinstance(y_targ, np.ndarray):
        y_targ = torch_to_numpy(y_targ)
    acc = np.sum(y_pred.argmax(1) == y_targ)
    return acc / y_targ.size

def mean_absolute_error(y_pred, y_targ):
    if not isinstance(y_pred, np.ndarray):
        y_pred = torch_to_numpy(y_pred)
    if not isinstance(y_targ, np.ndarray):
        y_targ = torch_to_numpy(y_targ)
    mae = np.mean(np.abs(y_pred - y_targ))
    return mae
