import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import settings
import utils
from Coral import CORAL


__all__ = ['ResNet', 'resnet50']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

resNet_main = False

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False) #True
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)  #True
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.resize = nn.ZeroPad2d(80)                                     ###
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False) #True
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.avgpool = nn.AvgPool2d(6, stride=1, padding=2)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        self.classifier = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, source, target=None):
        if (settings.image_size[0] < 224 or settings.image_size[1] < 224):
            source = self.resize(source)
            if target is not None:
                target = self.resize(target)

        coral_loss = []

        source = self.conv1(source)
        if target is not None:
            target = self.conv1(target)

        source = self.bn1(source)
        if target is not None:
            target = self.bn1(target)

        source = self.relu(source)
        if target is not None:
            target = self.relu(target)
            coral_loss.append(CORAL(source, target))

        source = self.maxpool(source)
        if target is not None:
            target = self.maxpool(target)

        source = self.layer1(source)
        if target is not None:
            target = self.layer1(target)
            coral_loss.append(CORAL(source, target))

        source = self.layer2(source)
        if target is not None:
            target = self.layer2(target)
            coral_loss.append(CORAL(source, target))

        source = self.layer3(source)
        if target is not None:
            target = self.layer3(target)
            coral_loss.append(CORAL(source, target))

        source = self.layer4(source)
        if target is not None:
            target = self.layer4(target)
            coral_loss.append(CORAL(source, target))

        source = self.avgpool(source)
        if target is not None:
            target = self.avgpool(target)

        source = source.view(source.size(0), -1)
        if target is not None:
            target = target.view(target.size(0), -1)
            coral_loss.append(CORAL(source, target))

        if resNet_main == True:
            source = self.classifier(source)

        return source, target, coral_loss

class DeepCoral(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepCoral, self).__init__()
        self.sharedNet = resnet50(settings.use_checkpoint)
        self.cls_fc = nn.Linear(2048, num_classes)

    def forward(self, source, target):
        source, target, coral_loss = self.sharedNet(source, target)
        source = self.sharedNet.classifier(source)

        return source, coral_loss


def resnet50(load):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if load == True:
        utils.load_net(model, "checkpoint.tar")
    return model