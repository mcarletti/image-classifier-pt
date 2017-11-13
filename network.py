import torch
import torchvision
import torch.nn.functional as F

class Network(torch.nn.Module):

    def __init__(self, input_shape, nb_classes):
        super(Network, self).__init__()

        self.features = torchvision.models.vgg16_bn(pretrained=True).features

        x = self.features(torch.autograd.Variable(torch.zeros(1, *input_shape)))
        self.nfts = x.numel()

        self.fc1 = torch.nn.Linear(self.nfts, 1024)
        self._xavier_init(self.fc1)

        self.fc1_bn = torch.nn.BatchNorm2d(1024)
        self._normal_init(self.fc1_bn, 0.0, 0.01)

        self.fc2 = torch.nn.Linear(1024, 1024)
        self._xavier_init(self.fc2)

        self.fc2_bn = torch.nn.BatchNorm2d(1024)
        self._normal_init(self.fc2_bn, 0.0, 0.01)

        self.fc3 = torch.nn.Linear(1024, nb_classes)
        self._normal_init(self.fc3, 0.0, 0.001)

    def forward(self, x):
        x = F.relu(self.features(x))
        x = x.view(-1, self.nfts)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3(x)
        return x

    def _normal_init(self, module, mu, std):
        module.weight.data.normal_(mu, std)
        module.bias.data.fill_(mu)
    
    def _xavier_init(self, module):
        import math
        if len(module.weight.data.shape) > 1:
            N_in = module.weight.data.size()[1]
            N_out = module.weight.data.size()[0]
            N = (N_in + N_out) / 2
        else:
            N = module.weight.data.size()[0]
        xavier_var = 1. / N
        xavier_std = math.sqrt(xavier_var)
        module.weight.data.normal_(0.0, xavier_std)
        module.bias.data.fill_(0.0)
