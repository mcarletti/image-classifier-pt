# -*- coding: utf-8 -*-

import argparse

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import os

from utils import *
from metrics import *
from dataset import DATASET

# Results computation and visualization -------

def test_image(model, image, use_cuda=False, gpu_id=0):
    '''Test one single image, which must be a PyTorch FloatTensor (CxHxW).
    '''
    model.eval()

    if use_cuda:
        image = image.cuda(gpu_id)

    pred = model(Variable(image.unsqueeze(0)))
    pred = torch_to_numpy(pred).squeeze()

    return pred

def test(model, dataloader, loss_fun=None, score_fun=None, use_cuda=False, gpu_id=0, hflip=False):
    '''Test the dataloader, one image at time. If \'loss_fun\' and/or \'score_fun\' are specified,
    also loss and score values are computed. If \'hflip\' is True, the prediction value of the
    i-th image is computed as the mean of the original image and its horizontal flipped
    version.
    '''
    
    model.eval()
    nb_samples = len(dataloader.dataset)
    predictions = []
    targets = []

    for i in range(nb_samples):
        image, target = dataloader.dataset[i]
        image.unsqueeze_(0)

        if hflip:
            flipped_image = image.numpy()[:,:,:,::-1].copy()
            image = torch.cat((image, torch.from_numpy(flipped_image)))

        if use_cuda:
            image = image.cuda(gpu_id)

        pred = model(Variable(image))
        pred = np.mean(torch_to_numpy(pred), axis=0).squeeze()
        
        predictions.append(pred)
        targets.append(target)

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    loss, score = None, None

    if loss_fun is not None:
        p = numpy_to_torch(predictions, use_cuda)
        t = numpy_to_torch(targets, False)
        if isinstance(t.data, torch.IntTensor):
            t = t.type(torch.LongTensor)
        if use_cuda:
            t = t.cuda()
        loss = loss_fun(p, t)
        loss = torch_to_numpy(loss)[0]

    if score_fun is not None:
        score = score_fun(predictions, targets)

    return predictions, targets, loss, score

def show_outliers(arr, title='Outliers', show_immediately=False):
    '''Extract and show the outliers in the input vector.
    '''
    N = arr.size
    xx = np.linspace(0, N - 1, N)
    out_idx = find_outliers(arr)

    plt.figure()
    plt.plot(xx, arr, color='r', label='data')
    plt.scatter(out_idx, arr[out_idx], color='b', label='outliers')
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if show_immediately:
        plt.show()

if __name__ == '__main__':

    seed = 23092017
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #%% Check arguments ------------------------------------------------------------

    # Parse arguments.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir', type=str, required=True,
                        help='results folder path')
    parser.add_argument('--dataset_dir', type=str, required=True,
                    help='dataset folder containing \'train.mat\' and \'test.mat\'')
    parser.add_argument('--input_shape', type=int, default=224,
                        help='input image resolution')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of samples per batch')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='enable cuda and use the specified gpu')
    parser.add_argument('--use_cuda', action='store_true',
                        help='enable cuda; if enabled and \'gpu_id\' is not set, the first available gpu will be used by default')
    parser.add_argument('--verbose', action='store_true',
                        help='print additional information')

    args = parser.parse_args()

    assert os.path.isdir(args.model_dir)

    if args.gpu_id is not None:
        args.use_cuda = True
    
    if args.use_cuda:
        if args.gpu_id is None:
            args.gpu_id = 0

    if args.verbose:
        print(args)

    #%% Load model -----------------------------------------------------------------

    mu = (0.485, 0.456, 0.406)
    sd = (0.229, 0.224, 0.225)

    # Normalize RGB-8 images between -1 and +1.
    transf = transforms.Compose([transforms.Resize((args.input_shape,args.input_shape)),
                                transforms.ToTensor(),
                                transforms.Normalize(mu, sd)])

    train_predictions = []
    train_targets = []
    test_predictions = []
    test_targets = []

    if args.verbose:
        print('Loading dataset:', args.dataset_dir)
    trainset = DATASET(parent=args.dataset_dir, train=True, transform=transf)
    testset = DATASET(parent=args.dataset_dir, train=False, transform=transf)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=1)  

    model_name = os.path.join(args.model_dir, 'model.pth')
    if args.verbose:
        print('Loading model:', model_name)
    model = torch.load(model_name)
    model.eval()

    if args.use_cuda:
        model.cuda(args.gpu_id)

    # Testing -------------------------

    try:
        if args.verbose:
            print('Testing...')
        train_predictions, train_targets, _, _ = test(model, trainloader, use_cuda=args.use_cuda, gpu_id=args.gpu_id)
        test_predictions, test_targets, _, _ = test(model, testloader, use_cuda=args.use_cuda, gpu_id=args.gpu_id)
    except Exception as e:
        #print(e.with_traceback, e.args)
        raise e

    #%% Compute final scores ---------------------------------------------------

    print('#' * 60)

    train_mae = accuracy(train_predictions, train_targets)
    print('Train Accuracy:', train_mae)

    test_mae = accuracy(test_predictions, test_targets)
    print('Test Accuracy:', test_mae)
