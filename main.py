from __future__ import print_function
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

import dataset

import matplotlib.pyplot as plt
import numpy as np


seed = 23092017

HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, using CPU')
    HAS_CUDA = False
else:
    torch.cuda.manual_seed_all(seed)
    gpu_id = 0


mu = (0.485, 0.456, 0.406)
sd = (0.229, 0.224, 0.225)
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, sd)])

train_set = dataset.DATASET(train=True, transform=transf)
#test_set = dataset.DATASET(train=False, transform=transf)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
#test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

image, target, label = train_set[np.random.randint(len(train_set))]

def imshow(img):
    for i in range(3):
        img[i, :, :] = img[i, :, :] * sd[i] + mu[i]
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))

imshow(image)
plt.show()

model = torchvision.models.vgg16_bn(pretrained=True)
if HAS_CUDA:
    model.cuda(gpu_id)
model.eval()

pred = model(torch.autograd.Variable(image.unsqueeze(0).cuda(gpu_id)))
pred = pred.data.cpu().numpy()

class_id = np.argmax(pred)

class_labels = np.loadtxt(open('data/ilsvrc_2012_labels.txt'), dtype=object, delimiter='\n')
print(class_id, class_labels[class_id])
