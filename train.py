# -*- coding: utf-8 -*-

import argparse

import torch
from torch.autograd import Variable
from torchvision.models import alexnet, inception_v3, vgg16, resnet50
import torchvision.transforms as transforms

import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from utils import *
from metrics import accuracy
from dataset import DATASET

from test import test

#%% Optimization ------------------------------

def train(model, dataloader, optimizer, loss_fun, score_fun, verbose=False):
    '''Train the model using dataloader for one epoch, using batches.
    The evaluation of the training is done at the end of the epoch.
    Data augmentation is performed as hard examples mining, horizontal flip
    and gaussian noise augmentation.
    '''
    
    model.train()
    nb_batches = len(dataloader)
    progress = ''

    # train on the entire dataset once
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if verbose:
            progress = '\b' * len(progress) + 'Progress > {:3d}'.format(int(100 * (batch_idx + 1) / nb_batches))
            print(progress, end='')
            sys.stdout.flush()

        # hirizontal flip (data augmentation)
        if args.augment_hflip:
            flipped_images = inputs.numpy()[:,:,:,::-1].copy()
            inputs = torch.cat((inputs, torch.from_numpy(flipped_images)))
            targets = torch.cat((targets, targets))
        
        # add noise (data augmentation)
        if args.augment_noise:
            noise_scale = 0.05
            images = inputs.numpy().copy()
            noise = (np.random.rand(*images.shape) - 0.5) * noise_scale
            noisy_images = (images + noise).astype(np.float32)
            inputs = torch.cat((inputs, torch.from_numpy(noisy_images)))
            targets = torch.cat((targets, targets))

        # prepare inputs
        if isinstance(targets, torch.IntTensor): # needed for classification
            targets = targets.type(torch.LongTensor)
        inputs, targets = Variable(inputs), Variable(targets)
        if args.use_cuda:
            inputs, targets = inputs.cuda(args.gpu_id), targets.cuda(args.gpu_id)

        # compute predictions
        outputs = model(inputs)
        loss = loss_fun(outputs, targets)

        # optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if verbose:
        print('')

    # compute epoch statistics
    _, _, epoch_loss, epoch_score = test(model, dataloader, loss_fun, score_fun, use_cuda=args.use_cuda, gpu_id=args.gpu_id)

    return epoch_loss, epoch_score

if __name__ == '__main__':

    seed = 23092017
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    models = {'alexnet': alexnet,   'googlenet': inception_v3, 'vgg': vgg16,     'resnet': resnet50}
    shapes = {'alexnet': (224,224), 'googlenet': (299,299),    'vgg': (224,224), 'resnet': (224,224)}

    #%% Check arguments ------------------------------------------------------------

    # Parse arguments.
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='resnet',
                        help='off-the-shelf pretrained model to fine-tune; must be in [\'alexnet\', \'vgg\', \'resnet\', \'googlenet\']')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='dataset folder containing \'train.mat\' and \'test.mat\'')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='output directory where to save saliency results')
    parser.add_argument('--epochs', type=int, default=4,
                        help='total number of iteration on the entire training set')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of samples per batch')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type [sgd, adam, rms]')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='optimizer learning rate coefficient')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='optimizer momentum coefficient')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='L2-regularizer coefficient')
    parser.add_argument('--augment_all', action='store_true',
                        help='data augmentation; enable all methods of data augmentation')
    parser.add_argument('--augment_hflip', action='store_true',
                        help='data augmentation by horizontal flip - WARNING: this doubles the batch size')
    parser.add_argument('--augment_noise', action='store_true',
                        help='data augmentation by adding gaussin noise - WARNING: this increase the batch size')
    parser.add_argument('--fine_tune_all', action='store_true',
                        help='enable fine-tuning of all the network architecture')
    parser.add_argument('--gpu_id', type=int, default=None,
                        help='enable cuda and use the specified gpu')
    parser.add_argument('--use_cuda', action='store_true',
                        help='enable cuda; if enabled and \'gpu_id\' is not set, the first available gpu will be used by default')
    parser.add_argument('--save_ckpt', action='store_true',
                        help='during training, save in \'ckpt\' folder the best state of the network according to testing results')
    parser.add_argument('--verbose', action='store_true',
                        help='print additional information')

    args = parser.parse_args()

    if args.augment_all:
        args.augment_hflip = True
        args.augment_hflip = True

    if args.gpu_id is not None:
        args.use_cuda = True
    
    if args.use_cuda:
        if args.gpu_id is None:
            args.gpu_id = 0

    if args.verbose:
        print(args)

    assert args.model in models.keys()
    assert args.optimizer in ['sgd', 'adam', 'rms']

    #%% Load data ------------------------------------------------------------------

    try:
        mu = (0.485, 0.456, 0.406)
        sd = (0.229, 0.224, 0.225)

        # Normalize RGB-8 images between -1 and +1.
        transf = transforms.Compose([transforms.Scale(shapes[args.model]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mu, sd)])

        trainset = DATASET(parent=args.dataset_dir, train=True, transform=transf)
        testset = DATASET(parent=args.dataset_dir, train=False, transform=transf)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=1)

        nb_classes = len(set(trainset.values))
    except:
        raise Exception('Unknown dataset name or path')

    #%% Load model -----------------------------------------------------------------

    # Set model.
    model = models[args.model](pretrained=True)
    model.train()

    enable_grads(model, args.fine_tune_all)

    if args.model in ['alexnet', 'vgg']:
        linear_layer_id = 0
        while type(model.classifier[linear_layer_id]) is not torch.nn.Linear:
            linear_layer_id += 1
        nfts = model.classifier[linear_layer_id].in_features
        model.classifier = torch.nn.Linear(nfts, nb_classes)
        normal_init(model.classifier)
        classifier = model.classifier
    elif args.model in ['googlenet', 'resnet']:
        nfts = model.fc.in_features
        model.fc = torch.nn.Linear(nfts, nb_classes)
        normal_init(model.fc)
        classifier = model.fc
    else:
        raise Exception('Unknown model name:', args.model)

    enable_grads(classifier, True)

    if args.fine_tune_all:
        model_to_train = model
    else:
        model_to_train = classifier

    #if os.path.exists('model.pth'):
    #    model = torch.load('model.pth')

    loss_fun = torch.nn.CrossEntropyLoss()
    score_fun = accuracy

    if args.use_cuda:
        model.cuda(args.gpu_id)
        loss_fun.cuda(args.gpu_id)

    if args.verbose:
        #print(model)
        print(model_to_train)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model_to_train.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model_to_train.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'rms':
        optimizer = torch.optim.RMSprop(model_to_train.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise Exception('Unknown optimizer:', args.optimizer)

    #%% Training -------------------------------------------------------------------

    def adjust_learning_rate(optimizer, epoch, step=1):
        """Sets the learning rate to the initial LR halved every 'step' epochs"""
        lr = args.learning_rate * (0.5 ** (epoch // step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    try:
        history = []
        best_test_score = float('Inf')
        
        nb_epochs = args.epochs
        for epoch in range(nb_epochs):
            print('#' * 60)
            print('Epoch [{:3d}|{:3d}]'.format(epoch+1, nb_epochs))

            adjust_learning_rate(optimizer, epoch, step=5)
            
            # train one epoch and evaluate the trainset at the end
            train_loss, train_score = train(model, trainloader, optimizer, loss_fun, score_fun, verbose=args.verbose)
            print('Train - Loss:', train_loss, 'Score:', train_score)

            # evaluate generalization
            _, _, test_loss, test_score = test(model, testloader, loss_fun, score_fun, use_cuda=args.use_cuda, gpu_id=args.gpu_id)
            print('Test -- Loss:', test_loss, 'Score:', test_score)
        
            # save partial results
            if args.save_ckpt and test_score < best_test_score:
                best_test_score = test_score
                ckpt_folder = os.path.join(args.output_dir, 'ckpt/')
                if not os.path.exists(ckpt_folder):
                    os.makedirs(ckpt_folder)
                os.remove(os.path.join(ckpt_folder, '*.ckpt'))
                torch.save(model, os.path.join(ckpt_folder, 'model_at_epoch_' + str(epoch + 1) + '.ckpt'))
            
            # update history
            history.append([[train_loss, train_score], [test_loss, test_score]])
        
        history = np.asarray(history)

        # create root folder for results
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # create final destination folder
        date = time.strptime(time.asctime())
        date = [date.tm_year, date.tm_yday, date.tm_hour, date.tm_min]
        date = ['{:03d}'.format(x) if i==1 else str(x) for i,x in enumerate(date)]
        date = ':'.join([x for x in date])
        dst_folder = date
        dst_folder = os.path.join(args.output_dir, dst_folder)
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        # save train info
        with open(os.path.join(dst_folder, 'info.txt'), 'w') as fp:
            info = str(vars(args))[1:-1]
            info = info.split(',')        
            info = '\n'.join(info)
            fp.write(info)

        # save final model
        torch.save(model, os.path.join(dst_folder, 'model.pth'))

        # compute, print and save final results
        print('#' * 60)
        print('Final results:')
        results = {}
        results['history'] = history
        preds, targets, loss, score = test(model, trainloader, loss_fun, score_fun, use_cuda=args.use_cuda, gpu_id=args.gpu_id)
        print('Train - Loss:', loss, 'Score:', score)
        results['train_preds'] = preds
        results['train_targets'] = targets
        preds, targets, loss, score = test(model, testloader, loss_fun, score_fun, use_cuda=args.use_cuda, gpu_id=args.gpu_id)
        print('Test -- Loss:', loss, 'Score:', score)
        results['test_preds'] = preds
        results['test_targets'] = targets

        # save numerical results and predictions
        #with open(os.path.join(dst_folder, 'results.pickle'), 'wb') as fp:
        #    pickle.dump(results, fp, pickle.HIGHEST_PROTOCOL)
        savemat(os.path.join(dst_folder, 'results.mat'), results)

        print('#' * 60)
        print('DONE - Result files saved in:', dst_folder)
        print('#' * 60)
        print('#' * 60)
        
    except Exception as e:
        #print(e.with_traceback, e.args)
        raise e
