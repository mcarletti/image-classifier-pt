from __future__ import print_function
import torch
import torchvision
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# check cuda support
HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, using CPU')
    HAS_CUDA = False

print('Load image')
image = np.asarray(Image.open('data/sample.jpg'))
assert(image.shape == (224, 224, 3))

plt.imshow(image)
plt.show()

print('Loading pretrained model')
model = torchvision.models.vgg16_bn(pretrained=True)
model.eval()
if HAS_CUDA:
    model.cuda()

print('Image preprocessing')
mu = (0.485, 0.456, 0.406)
sd = (0.229, 0.224, 0.225)
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mu, sd)])

image = transf(image).unsqueeze(0)
image = torch.autograd.Variable(image)
if HAS_CUDA:
    image = image.cuda()

print('Computing prediction')
pred = model(image)
pred = pred.data.cpu().numpy()

# show results
class_id = np.argmax(pred)
class_labels = np.loadtxt(open('data/ilsvrc_2012_labels.txt'), dtype=object, delimiter='\n')
print(class_labels[class_id])
