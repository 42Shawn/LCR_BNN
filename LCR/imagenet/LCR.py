import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.stats import norm
import scipy

import math


def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)


class LCR(nn.Module):
    def __init__(self, fp_net, net):
        super(LCR, self).__init__()

        fp_channels = fp_net.get_channel_num()
        channels = net.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(fp_channels, channels)])

        bns = fp_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.fp_net = fp_net
        self.net = net

    def retention_matrix(self, fm1, fm2):
        if fm1.size(2) > fm2.size(2):
            fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

        fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
        fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1, 2)

        rm = torch.bmm(fm1, fm2) / fm1.size(2)
        return rm

    def TopEigenvalue(self, K, n_power_iterations=1, dim=1):
        v = torch.ones(K.shape[0], K.shape[1], 1).to('cuda')
        for _ in range(n_power_iterations):
            m = torch.bmm(K, v)
            n = torch.norm(m, dim=1).unsqueeze(1)
            v = m / n

        value = torch.sqrt(n / torch.norm(v, dim=1).unsqueeze(1))
        return value

    def forward(self, x):

        fp_feats = self.fp_net.extract_feature(x, preReLU=False)
        binary_feats = self.net.scaleable_feature(x, preReLU=False)

        feat_num = len(fp_feats)

        loss_lcr = 0
        for i in range(feat_num):

            if i < feat_num - 1:

                RM_b = torch.bmm(self.retention_matrix(binary_feats[i], binary_feats[i + 1]), self.retention_matrix(binary_feats[i], binary_feats[i + 1]).transpose(2,1)) #for calculate the eigenvalue of retention matrix, because retention matrix is asymmetric
                RM_fp = torch.bmm(self.retention_matrix(fp_feats[i].detach(), fp_feats[i + 1].detach()), self.retention_matrix(fp_feats[i].detach(), fp_feats[i + 1].detach()).transpose(2,1))

                loss_lcr += F.mse_loss(self.TopEigenvalue(K=RM_b), self.TopEigenvalue(K=RM_fp)) / 2 ** (feat_num - i - 1)

        return loss_lcr
