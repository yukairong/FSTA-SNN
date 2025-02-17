import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from models.freq_layer import DCTSA

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

class freqLIF(nn.Module):
    def __init__(self, step, channel, h, w):
        super(freqLIF, self).__init__()
        self.step = step
        self.temp = 3.0
        self.v_th = 1.0
        self.grad_scale = 0.1
        self.ffilterList = nn.ModuleList(
            [fftFilter(channel, h, w)
            for i in range(self.step)]
        )
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[0].numel() * self.step)
        mem = torch.zeros_like(x[0])
        out = []
        for i in range(self.step):
            mem = 0.25 * mem + x[i]
            # mem = self.bn(mem)
            mem = self.ffilterList[i](mem)
            spike = spike_activation(mem / self.v_th)
            mem = mem * (1 - spike)

            out += [spike]

        out = torch.stack(out)

        return out

class DctLIF(nn.Module):
    def __init__(self, step, channel, h, w):
        super(DctLIF, self).__init__()
        self.step = step
        self.temp = 3.0
        self.v_th = 1.0
        self.grad_scale = 0.1
        # self.bn = nn.BatchNorm2d(channel)
        # self.DctSpatialLayer = DctSpatialFilter()
        # self.DctChannelLayer = DctChannelFilter(c=channel, reduction=16)
        self.MemMemory = MemMLP(d_model=channel, hidden=8, drop_prob=0.)
        # self.fusion = nn.Conv2d(channel * 2, channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.layernorm = nn.LayerNorm([channel, h, w])
        self.dsa = FSA_DCT(reduction=1, dimension=channel, h=h, w=w)

    
    def forward(self, x):
        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[0].numel() * self.step)
        # mem = torch.zeros_like(x[0])
        mem = self.MemMemory(x)
        out = []
        for i in range(self.step):
            mem = 0.25 * mem + x[i]
            # mem = self.layernorm(mem)
            # mem_c = self.DctChannelLayer(mem)
            # mem_s = self.DctSpatialLayer(mem)
            mem = self.dsa(mem)
            # mem = torch.cat([mem_c, mem_s], dim=1)
            # mem = self.fusion(mem)
            # mem = self.bn(mem)

            spike = spike_activation(mem / self.v_th)
            mem = mem * (1 - spike)

            out += [spike]

        out = torch.stack(out)
        return out

# 切割CA版本
# class DctSpatialLIF(nn.Module):
#     def __init__(self, step, channel, h, w, freq_num, reduction):
#         super(DctSpatialLIF, self).__init__()
#         self.step = step
#         self.temp = 3.0
#         self.v_th = 1.0
#         self.grad_scale = 0.1
#         self.bn = nn.BatchNorm2d(channel)
#         self.ta = nn.ModuleList([CA(channel=channel) for _ in range(step)])
#         if h >= int(math.sqrt(freq_num)):
#             self.dct_spatial = DctSA(freq_num=freq_num, channel=channel, h=h, w=w, reduction=reduction, select_method='all')
#         else:
#             # self.dct_spatial = DCT_Spatial(freq_num=9, channel=channel, h=h, w=w, reduction=1, select_method='all')
#             self.dct_spatial = nn.Identity()
    
#     def forward(self, x):
#         if self.grad_scale is None:
#             self.grad_scale = 1 / math.sqrt(x[0].numel() * self.step)
#         mem = torch.zeros_like(x[0])
#         # mem = self.MemMemory(x)
#         out = []
#         for i in range(self.step):
#             mem = 0.25 * mem + x[i]
#             mem, weight = self.ta[i](mem)
#             if isinstance(self.dct_spatial, DctSA):
#                 mem = self.dct_spatial(mem, weight)
#             else:
#                 mem = self.bn(mem)
#             spike = spike_activation(mem / self.v_th)
#             mem = mem * (1 - spike)

#             out += [spike]
#         out = torch.stack(out)

#         return out

class DctSpatialLIF(nn.Module):
    def __init__(self, step, channel, freq_num, reduction, h=None, w=None):
        super(DctSpatialLIF, self).__init__()
        self.step = step
        self.temp = 3.0
        self.v_th = 1.0
        self.grad_scale = 0.1
        self.dct = DCTSA(freq_num=freq_num, channel=channel, step=step, reduction=reduction, groups=1, select_method='all')
        # self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        if self.grad_scale is None:
            self.grad_scale = 1 / math.sqrt(x[0].numel() * self.step)
        mem = torch.zeros_like(x[0])
        out = []
        x = self.dct(x)
        for i in range(self.step):
            mem = 0.25 * mem + x[i]
            spike = spike_activation(mem / self.v_th)
            mem = mem * (1 - spike)

            out += [spike]
        out = torch.stack(out)

        return out
    
    

class SpikeConv(nn.Module):

    def __init__(self, conv, step=2):
        super(SpikeConv, self).__init__()
        self.conv = conv
        self.step = step

    def forward(self, x):
        out = []
        for i in range(self.step):
            out += [self.conv(x[i])]
        out = torch.stack(out)

        return out
    

class SpikePool(nn.Module):

    def __init__(self, pool, step=2):
        super(SpikePool, self).__init__()
        self.pool = pool
        self.step = step

    def forward(self, x):
        T, B, C, H, W = x.shape
        out = x.reshape(-1, C, H, W)
        out = self.pool(out)
        B_o, C_o, H_o,  W_o = out.shape
        out = out.view(T, B, C_o, H_o, W_o).contiguous()

        return out
    
class myBatchNorm3d(nn.Module):
    
    def __init__(self, bn, step=2):
        super(myBatchNorm3d, self).__init__()
        self.bn = nn.BatchNorm3d(bn.num_features)
        self.step = step

    def forward(self, x):
        out = x.permute(1, 2, 0, 3, 4)
        out = self.bn(out)
        out = out.permute(2, 0, 1, 3, 4).contiguous()

        return out
    
    
        