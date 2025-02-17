import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange
from data.dct_filter import DCT8x8, DCT7x7, DCT3x3

def spike_activation(x, ste=False, temp=1.0):
    out_s = torch.gt(x, 0.5)
    if ste:
        out_bp = torch.clamp(x, 0, 1)
    else:
        out_bp = torch.clamp(x, 0, 1)
        out_bp = (torch.tanh(temp * (out_bp - 0.5)) + np.tanh(temp * 0.5)) / (2 * (np.tanh(temp * 0.5)))
    return (out_s.float() - out_bp).detach() + out_bp


def MPR(s, thresh):
    s[s > 1.] = s[s > 1.] ** (1.0 / 3)
    s[s < 0.] = -(-(s[s < 0.] - 1.)) ** (1.0 / 3) + 1.
    s[(0. < s) & (s < 1.)] = 0.5 * torch.tanh(3. * (s[(0. < s) & (s < 1.)] - thresh)) / np.tanh(3. * (thresh)) + 0.5

    return s

def gradient_scale(x, scale):
    yout = x
    ygrad = x * scale
    y = (yout - ygrad).detach() + ygrad
    
    return y

def mem_update(bn, x_in, mem, v_th, decay, grad_scale=1., temp=1.0):
    mem = mem * decay + x_in
    mem_bn = mem
    
    spike = spike_activation(mem_bn / v_th, temp=temp)
    mem = mem * (1 - spike)
    
    return mem, spike


class LIFAct(nn.Module):

    def __init__(self, step, channel):
        super(LIFAct, self).__init__()
        self.step = step
        self.v_th = 1.0
        self.temp = 3.0
        self.grad_scale = 0.1
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[0].numel() * self.step)
        u = torch.zeros_like(x[0])
        out = []
        for i in range(self.step):
            u, out_i = mem_update(bn=self.bn, x_in=x[i], mem=u, v_th=self.v_th, 
                                  grad_scale=self.grad_scale, decay=0.25, temp=self.temp)
            out += [out_i]

        out = torch.stack(out)
        return out

class FreConv(nn.Module):
    def __init__(self, c, reduction, k=1, p=0):
        super(FreConv, self).__init__()
        if reduction == 1:
            self.freq_attention = nn.Sequential(
                nn.Conv2d(c, 1, kernel_size=k, padding=p, bias=False),
            )
        else:
            self.freq_attention = nn.Sequential(
                nn.Conv2d(c, c // reduction, kernel_size=k, bias=False, padding=p),
                nn.ReLU(),
                nn.Conv2d(c // reduction, 1, kernel_size=k, padding=p, bias=False)
                )
            

    def forward(self, x):

        return self.freq_attention(x)
    

class DCTSA(nn.Module):
    def __init__(self, freq_num, channel, step, reduction=1, groups=1, select_method='all'):
        super(DCTSA, self).__init__()
        self.freq_num = freq_num
        self.channel = channel
        self.reduction = reduction
        self.select_method = select_method
        self.groups = groups
        self.step = step

        if freq_num == 64:
            self.dct_filter = DCT8x8()
            self.p = int((self.dct_filter.freq_range - 1)/ 2)
        elif freq_num == 49:
            self.dct_filter = DCT7x7()
            self.p = int((self.dct_filter.freq_range - 1)/ 2)
        elif freq_num == 9:
            self.dct_filter = DCT3x3()
            self.p = int((self.dct_filter.freq_range - 1) / 2)

        if self.select_method == 'all':
            self.dct_c = self.dct_filter.freq_num
        elif 's' in self.select_method:
            self.dct_c = 1
        elif 'top' in self.select_method:
            self.dct_c = int(self.select_method.replace('top', ''))

        self.freq_attention = FreConv(self.dct_c, reduction=reduction, k=7, p=3)
        self.sigmoid = nn.Sigmoid()
        
        # cahnnel select
        self.avg_pool_c = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.max_pool_c = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.register_parameter('alpha', nn.Parameter(torch.FloatTensor([0.5])))
        self.register_parameter('beta', nn.Parameter(torch.FloatTensor([0.5])))

        
        # self.fc_c = nn.Linear(channel, channel, bias=False)
        self.fc_t = nn.Linear(step, step, bias=False)

        self.register_parameter('t', nn.Parameter(torch.FloatTensor([0.6])))    # m
        self.register_parameter('s', nn.Parameter(torch.FloatTensor([0.5])))  # n
        self.register_parameter('x', nn.Parameter(torch.FloatTensor([1])))

        self.register_parameter('t_scale', nn.Parameter(torch.FloatTensor([1])))
        self.register_parameter('s_scale', nn.Parameter(torch.FloatTensor([1])))

    def forward(self, x):
        t, b, c, h, w = x.shape
        x = rearrange(x, 't b c h w -> b t c h w')
        avg_map = self.avg_pool_c(x)    # (b, t, c, 1, 1)
        max_map = self.max_pool_c(x)

        map_add = self.alpha * avg_map + self.beta * max_map

        # time branch
        # map_fusion_t = self.fc_t(map_add)   # (b, t, c, 1, 1)
        map_add = rearrange(map_add, 'b t c 1 1 -> b c t')
        # map_fusion_t = self.fc_t(map_add.squeeze().transpose(1, 2)).transpose(1, 2) # (b, c, t) -> (b, t, c)
        map_fusion_t = self.fc_t(map_add).transpose(1, 2) # (b, c, t) -> (b, t, c)

        ## time
        t_mean_sig = self.sigmoid(torch.mean(map_fusion_t, dim=2))    # (b, t)
        t_mean_sig = rearrange(t_mean_sig, 'b t -> b t 1 1 1')
        t_mean_sig = t_mean_sig.repeat(1, 1, c, h, w)
        x_t = x * t_mean_sig + x    # (b, t, c, h, w)

        ## sptial
        if self.select_method == 'all':
            dct_weight = self.dct_filter.filter

            dct_weight = dct_weight.unsqueeze(1)
            dct_weight = dct_weight.repeat(1, self.channel, 1, 1)   # * self.step

        elif 's' in self.select_method:
            filter_id = int(self.select_method.replace('s', ''))
            dct_weight = self.dct_filter.get_filter(filter_id)

            dct_weight = dct_weight.unsqueeze(0).unsqueeze(0)

            dct_weight = dct_weight.repeat(1, self.channel, 1, 1)

        elif 'top' in self.select_method:
            filter_id = self.dct_filter.get_topk(self.dct_c)
            dct_weight = self.dct_filter.get_filter(filter_id)
            dct_weight = dct_weight.unsqueeze(1)
            dct_weight = dct_weight.repeat(1, self.channel, 1, 1)

        dct_bias = torch.zeros(self.dct_c).to(dct_weight.device)
        dct_feature = F.conv2d(torch.mean(x_t, dim=1), dct_weight, dct_bias, stride=1, padding=self.p) # (b, dct_c, h, w)
        dct_feature = self.freq_attention(dct_feature)  # （b, 1, h, w)

        dct_feature = dct_feature.unsqueeze(1)  # (b, 1, 1, h, w)
        dct_feature = dct_feature.repeat(1, t, c, 1, 1) # (b, t, c, h, w)
        x_s = x_t * self.sigmoid(dct_feature) + x_t

        x = (x_t * self.t + x_s * self.s) / 2
        x = rearrange(x, 'b t c h w -> t b c h w')

        return x
