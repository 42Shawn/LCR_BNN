import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
from modules import BinarizeConv2d

BN = None

__all__ = ['resnet18_1w1a_IR', 'resnet34_1w1a_IR']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


def conv3x3Binary(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3Binary(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = conv3x3Binary(planes, planes)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        
        out = self.nonlinear(out)
        
        residual = out
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.nonlinear(out)
        
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_stem=False,
                 avg_down=False, bypass_last_bn=False,
                 bn_group_size=1,
                 bn_group=None,
                 bn_sync_stats=False,
                 use_sync_bn=True):

        global BN, bypass_bn_weight_list

        BN = nn.BatchNorm2d

        bypass_bn_weight_list = []


        self.inplanes = 64
        super(ResNet, self).__init__()

        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.gamma = nn.Parameter(torch.ones([1, 4]), requires_grad=True)

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                        BN(32),
                        nn.Hardtanh(inplace=True),
                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                        BN(32),
                        nn.Hardtanh(inplace=True),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(64)
        self.nonlinear1 = nn.Hardtanh(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.bn2 = nn.BatchNorm1d(512 * block.expansion)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            #intialize BN layer with a mean of zero and a variance of one
            #refer: https://forums.fast.ai/t/how-is-batch-norm-initialized/39818
            elif isinstance(m, nn.BatchNorm2d):
                # m.weight.data.fill_(1e-8)
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if bypass_last_bn:
            for param in bypass_bn_weight_list:
                param.data.zero_()
            print('bypass {} bn.weight in BottleneckBlocks'.format(len(bypass_bn_weight_list)))

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    BN(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    BN(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear1(x) 
        x = self.maxpool(x)     

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)      
        x = self.fc(x)

        return x

    def get_bn_before_relu(self):

        bn1 = self.layer1[-1].bn2
        bn2 = self.layer2[-1].bn2
        bn3 = self.layer3[-1].bn2
        bn4 = self.layer4[-1].bn2

        return [bn1, bn2, bn3, bn4]

    def get_channel_num(self):

        return [256, 512, 1024, 2048]

    def extract_feature(self, x, preReLU=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear1(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        if not preReLU:
            feat1 = F.hardtanh(feat1)
            feat2 = F.hardtanh(feat2)
            feat3 = F.hardtanh(feat3)
            feat4 = F.hardtanh(feat4)

        return [feat1, feat2, feat3, feat4]

    def scaleable_feature(self, x, preReLU=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.nonlinear1(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        if not preReLU:
            feat1 = F.hardtanh(feat1)
            feat2 = F.hardtanh(feat2)
            feat3 = F.hardtanh(feat3)
            feat4 = F.hardtanh(feat4)

        feat1 = self.gamma[0,0] * feat1
        feat2 = self.gamma[0,1] * feat2
        feat3 = self.gamma[0,2] * feat3
        feat4 = self.gamma[0,3] * feat4

        return [feat1, feat2, feat3, feat4]


def resnet18_1w1a_IR(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34_1w1a_IR(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

