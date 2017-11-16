from __future__ import print_function
import torch
from torchvision import transforms

import dataset
import network

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import os

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_image', type=str, default=None)
args = parser.parse_args()

HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, using CPU')
    HAS_CUDA = False

mu = (0.485, 0.456, 0.406)
sd = (0.229, 0.224, 0.225)
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, sd)])

print('Creating network')
net = torch.load('trained_model.pth')
net.eval()
if HAS_CUDA:
    net.cuda()

if args.input_image is not None:

    assert os.path.exists(args.input_image)

    print('Load image')
    image = np.asarray(Image.open(args.input_image).resize((100, 100)))
    assert(image.shape == (100, 100, 3))

    plt.imshow(image)
    plt.show()

    image = transf(image).unsqueeze(0)
    image = torch.autograd.Variable(image)
    if HAS_CUDA:
        image = image.cuda()

    print('Computing prediction')
    pred = net(image)
    pred = pred.data.cpu().numpy()

    class_id = np.argmax(pred)
    class_labels = np.loadtxt(open('data/bin_data/class_labels.txt'), dtype=object, delimiter='\n')
    print(class_labels[class_id])

else:

    print('Loading dataset')
    test_set = dataset.DATASET(train=False, transform=transf)
    class_names = test_set.labels
    nb_classes = class_names.shape[0]

    batch_size = 16
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

    print('Computing accuracy on testing set')
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

