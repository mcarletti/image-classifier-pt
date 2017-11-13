from __future__ import print_function
import torch
from torchvision import transforms

import dataset
import network

import matplotlib.pyplot as plt
import numpy as np


seed = 23092017

HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, using CPU')
    HAS_CUDA = False
    torch.manual_seed_all(seed)
else:
    torch.cuda.manual_seed_all(seed)
    gpu_id = 0

np.random.seed(seed)

mu = (0.485, 0.456, 0.406)
sd = (0.229, 0.224, 0.225)
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, sd)])

print('Loading dataset')
train_set = dataset.DATASET(train=True, transform=transf)
class_names = train_set.labels
nb_classes = class_names.shape[0]

batch_size = 24
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

use_data_augmentation_hflip = True


print('Creating network')
net = network.Network((3, 100, 100), nb_classes)
net.train()
if HAS_CUDA:
    net.cuda(gpu_id)

criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(net.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0005)
optimizer = torch.optim.Adam(net.parameters(), lr=0.05, weight_decay=0.0005)

if HAS_CUDA:
    criterion.cuda(gpu_id)

def lr_scheduler(optimizer, lr_decay=0.001, epoch=None, step=1):
    if epoch is None or step == 1 or (epoch+1) % step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= (1 - lr_decay)
    return optimizer

print('Start training')
nb_epochs = 20
nb_batches = len(train_loader)
for epoch in range(nb_epochs):
    for batch_id, data in enumerate(train_loader):
        images, targets, _ = data

        if use_data_augmentation_hflip:
            flipped_images = images.numpy()[:,:,:,::-1].copy()
            images = torch.cat((images, torch.from_numpy(flipped_images)))
            targets = torch.cat((targets, targets))

        if HAS_CUDA:
            images, targets = images.cuda(gpu_id), targets.type(torch.LongTensor).cuda(gpu_id)

        images, targets = torch.autograd.Variable(images), torch.autograd.Variable(targets)

        predictions = net(images)
        loss = criterion(predictions, targets)

        if abs(loss.data[0]) == float('Inf') or loss.data[0] is float('NaN'):
            print('EARLY STOP because of invalid loss value')
            valid_training = False
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('EPOCH - [%2d/%2d] - loss: %.3f' % (epoch + 1, nb_epochs, loss.data[0]))
    optimizer = lr_scheduler(optimizer, 0.25)


print('Computing accuracy on training set')
net.eval()
correct = 0
total = 0
for data in train_loader:
    images, targets, _ = data

    if HAS_CUDA:
        images, targets = images.cuda(gpu_id), targets.type(torch.LongTensor).cuda(gpu_id)

    predictions = net(torch.autograd.Variable(images, requires_grad=False))
    _, predicted = torch.max(predictions.data, 1)
    total += batch_size
    correct += (predicted == targets).sum()

print('Accuracy of the network on the training images: %d %%' % (100 * correct / total))

torch.save(net, 'trained_model.pth')
