from __future__ import print_function
import torch
from torchvision import transforms

import dataset
import network

import matplotlib.pyplot as plt
import numpy as np


HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, using CPU')
    HAS_CUDA = False

mu = (0.485, 0.456, 0.406)
sd = (0.229, 0.224, 0.225)
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, sd)])

print('Loading dataset')
test_set = dataset.DATASET(train=False, transform=transf)
class_names = test_set.labels
nb_classes = class_names.shape[0]

batch_size = 48
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)


print('Creating network')
net = network.Network((3, 100, 100), nb_classes)
net = torch.load('trained_model.pth')
if HAS_CUDA:
    net.cuda()


print('Computing accuracy on testing set')
net.eval()
correct = 0
total = 0
for data in test_loader:
    images, targets, _ = data

    if HAS_CUDA:
        images, targets = images.cuda(), targets.type(torch.LongTensor).cuda()

    predictions = net(torch.autograd.Variable(images, requires_grad=False))
    _, predicted = torch.max(predictions.data, 1)
    total += batch_size
    correct += (predicted == targets).sum()

print('Accuracy of the network on the testing images: %d %%' % (100 * correct / total))

