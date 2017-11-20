import torch
import torchvision
import torch.nn.functional as F


class INet(torch.nn.Module):

    def __init__(self, input_shape, nb_classes):
        super(INet, self).__init__()

        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.nfts = -1
        self.features = None
        self.classifier = None

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.nfts)
        x = self.classifier(x)
        return x

    def _normal_init(self, module, mu, std):
        class_name =  module.__class__.__name__
        for module_name in ['ReLU', 'MaxPool', 'Sequential']:
            if class_name.find(module_name) != -1:
                return
        module.weight.data.normal_(mu, std)
        module.bias.data.fill_(mu)
    
    def _xavier_init(self, module):
        class_name =  module.__class__.__name__
        for module_name in ['ReLU', 'MaxPool', 'Sequential']:
            if class_name.find(module_name) != -1:
                return
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


class Network(INet):

    def __init__(self, input_shape, nb_classes):
        INet.__init__(self, input_shape, nb_classes)

        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(self.input_shape[0], 32, 5),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(64, 48, 3),
            torch.nn.BatchNorm2d(48),
            torch.nn.ReLU(inplace=True), torch.nn.MaxPool2d(2,2))

        self.features.apply(self._xavier_init)

        x = self.features(torch.autograd.Variable(torch.zeros(1, *self.input_shape)))
        self.nfts = x.numel()

        self.fc1 = torch.nn.Linear(self.nfts, 1024)
        self._xavier_init(self.fc1)

        self.fc1_bn = torch.nn.BatchNorm2d(1024)
        self._normal_init(self.fc1_bn, 0.0, 0.01)

        self.fc2 = torch.nn.Linear(1024, 1024)
        self._xavier_init(self.fc2)

        self.fc2_bn = torch.nn.BatchNorm2d(1024)
        self._normal_init(self.fc2_bn, 0.0, 0.01)

        self.fc3 = torch.nn.Linear(1024, self.nb_classes)
        self._normal_init(self.fc3, 0.0, 0.001)

        self.classifier = torch.nn.Sequential(
            self.fc1, self.fc1_bn, torch.nn.ReLU(inplace=True),
            self.fc2, self.fc2_bn, torch.nn.ReLU(inplace=True),
            self.fc3)


class MobileNet(INet):

    def __init__(self, input_shape, nb_classes):
        INet.__init__(self, input_shape, nb_classes)

        def conv_bn(ch_inp, ch_out, stride):
            return torch.nn.Sequential(
                torch.nn.Conv2d(ch_inp, ch_out, 3, stride, 1, bias=False),
                torch.nn.BatchNorm2d(ch_out),
                torch.nn.ReLU(inplace=True)
            )

        def conv_dw(ch_inp, ch_out, stride):
            return torch.nn.Sequential(
                torch.nn.Conv2d(ch_inp, ch_inp, 3, stride, 1, groups=ch_inp, bias=False),
                torch.nn.BatchNorm2d(ch_inp),
                torch.nn.ReLU(inplace=True),
    
                torch.nn.Conv2d(ch_inp, ch_out, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(ch_out),
                torch.nn.ReLU(inplace=True),
            )

        ch = self.input_shape[0]

        self.features = torch.nn.Sequential(
            conv_bn(  ch,   32, 2), 
            conv_dw(  32,   64, 1),
            conv_dw(  64,  128, 2),
            conv_dw( 128,  128, 1),
            conv_dw( 128,  256, 2),
            conv_dw( 256,  256, 1),
            conv_dw( 256,  512, 2),
            conv_dw( 512,  512, 1),
            conv_dw( 512,  512, 1),
            conv_dw( 512,  512, 1),
            conv_dw( 512,  512, 1),
            conv_dw( 512,  512, 1),
            conv_dw( 512, 1024, 2),
            conv_dw(1024, 1024, 1),
            torch.nn.AvgPool2d(4))

        x = self.features(torch.autograd.Variable(torch.zeros(1, *self.input_shape)))
        self.nfts = x.numel()

        self.classifier = torch.nn.Linear(self.nfts, self.nb_classes)
