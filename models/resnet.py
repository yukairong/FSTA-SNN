import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from timm.models.registry import register_model
from timm.models import create_model

from models.spike_layer import SpikeConv, SpikePool, myBatchNorm3d, LIFAct, DctSpatialLIF

def conv3x3(in_planes, out_planes, stride=1):
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, step=2, freq_num=64, reduction=16):
        super(BasicBlock, self).__init__()
        c2wh = dict([(64, 16), (128, 8), (256, 4), (512, 2)])

        self.conv1 = SpikePool(pool=conv3x3(inplanes, planes, stride), step=step)
        self.bn1 = myBatchNorm3d(bn=nn.BatchNorm2d(planes), step=step)
        self.relu1 = DctSpatialLIF(step=step, channel=planes, h=c2wh[planes], w=c2wh[planes], freq_num=freq_num, reduction=reduction)
        self.conv2 = SpikePool(pool=conv3x3(planes, planes), step=step)
        self.bn2 = myBatchNorm3d(bn=nn.BatchNorm2d(planes), step=step)
        self.downsample = downsample
        self.stride = stride
        if downsample is None:
            self.relu2 = DctSpatialLIF(step=step, channel=planes, h=c2wh[planes], w=c2wh[planes], freq_num=freq_num, reduction=reduction)
            # self.relu2 = freqLIF(step=step, channel=planes, h=h, w=w)
        else:
            c = downsample[-1].bn.num_features
            self.relu2 = DctSpatialLIF(step=step, channel=c, h=c2wh[c], w=c2wh[c], freq_num=freq_num, reduction=reduction)
            # self.relu2 = freqLIF(step=step, channel=downsample[-1].bn.num_features, h=h, w=w)
            
    def forward(self, s):
        temp, x = s
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out1 = self.relu2(out.clone())

        return out, out1


class ResNet19(nn.Module):

    def __init__(self, block, layers, input_c=3, rp=False, step=2, num_classes=10, freq_num=64, reduction=16, pretrained_cfg=None, pretrained_cfg_overlay=None):
        super(ResNet19, self).__init__()

        self.rp = rp
        self.step = step
        
        inplanes = 128
        self.inplanes = inplanes
        self.conv1 = SpikePool(pool=nn.Conv2d(input_c, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
                               step=self.step)
        self.bn1 = myBatchNorm3d(bn=nn.BatchNorm2d(self.inplanes), step=self.step)
        self.relu = DctSpatialLIF(step=self.step, channel=self.inplanes, freq_num=freq_num, reduction=reduction)
        self.layer1 = self._make_layer(block, inplanes, layers[0], freq_num=freq_num)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2, freq_num=freq_num)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2, freq_num=freq_num)
        self.avgpool = SpikePool(pool=nn.AdaptiveAvgPool2d(1), step=self.step)
        self.fc = SpikeConv(conv=nn.Linear(inplanes * 4 * block.expansion, num_classes), step=step)

        for m in self.modules():
            if isinstance(m, SpikePool) and isinstance(m.pool, nn.Conv2d):
                n = m.pool.kernel_size[0] * m.pool.kernel_size[1] * m.pool.out_channels
                m.pool.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, myBatchNorm3d):
                m.bn.weight.data.fill_(1)
                m.bn.bias.data.zero_()
            elif isinstance(m, SpikeConv):
                n = m.conv.weight.size(1)
                m.conv.weight.data.normal_(0, 1.0/float(n))
                m.conv.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, freq_num=49, h=None, w=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SpikePool(
                    pool=nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    step=self.step
                ),
                myBatchNorm3d(bn=nn.BatchNorm2d(planes * block.expansion), step=self.step)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, step=self.step, freq_num=freq_num))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, step=self.step, freq_num=freq_num))

        return nn.Sequential(*layers)
    
    def forward(self, x, is_adain=False, is_drop=False, feat=None):

        if len(x.shape) == 4:
            x = x.repeat(self.step, 1, 1, 1, 1)
        else:
            x = x.permute([1, 0, 2, 3, 4])
        
        if not feat is None:
            x = self.fc(feat)
            
            return x
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            temp, x = self.layer1((x, x))
            temp, x = self.layer2((temp, x))
            temp, x = self.layer3((temp, x))
            if is_drop:
                temp = F.relu(temp)
                x = self.avgpool(temp)
            else:
                x = self.avgpool(x)
            
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
                fea = x
            elif len(x.shape) == 5:
                x = x.view(x.size(0), x.size(1), -1)
                fea = x.mean([0])
            x = self.fc(x)

            if len(x.shape) == 3:
                x = x.mean([0])
            if is_adain:
                return fea, x
            else:
                return x

class ResNet20(nn.Module):

    def __init__(self, block, layers, pretrained_cfg=None, pretrained_cfg_overlay=None, num_classes=10, input_c=3, rp=False, step=2, freq_num=64, reduction=16):
        super(ResNet20, self).__init__()

        global BN
        BN = nn.BatchNorm2d
        global ReLU
        ReLU = nn.ReLU

        self.rp = rp
        self.step = step
        
        self.inplanes = 64

        self.conv1 = nn.Sequential(
            SpikePool(nn.Conv2d(input_c, 64, kernel_size=3, stride=1, padding=1, bias=False), step=self.step),
            myBatchNorm3d(BN(64), step=self.step),
            DctSpatialLIF(step=self.step, channel=64, h=32, w=32, freq_num=freq_num, reduction=reduction),
            SpikePool(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), step=self.step),
            myBatchNorm3d(BN(64), step=self.step),
            DctSpatialLIF(step=self.step, channel=64, h=32, w=32, freq_num=freq_num, reduction=reduction),
            SpikePool(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), step=self.step),
            myBatchNorm3d(BN(64), step=self.step),
            DctSpatialLIF(step=self.step, channel=64, h=32, w=32, freq_num=freq_num, reduction=reduction)
        )
        self.avgpool = SpikePool(pool=nn.AvgPool2d(2), step=self.step)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, freq_num=freq_num, reduction=reduction)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, freq_num=freq_num, reduction=reduction)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, freq_num=freq_num, reduction=reduction)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, freq_num=freq_num, reduction=reduction)
        self.avgpool2 = SpikePool(pool=nn.AdaptiveAvgPool2d(1), step=self.step)
        self.fc = SpikeConv(conv=nn.Linear(512, num_classes, bias=True), step=self.step)

        for m in self.modules():
            if isinstance(m, SpikePool) and isinstance(m.pool, nn.Conv2d):
                n = m.pool.kernel_size[0] * m.pool.kernel_size[1] * m.pool.out_channels
                m.pool.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, myBatchNorm3d):
                m.bn.weight.data.fill_(1)
                m.bn.bias.data.zero_()
            elif isinstance(m, SpikeConv):
                n = m.conv.weight.size(1)
                m.conv.weight.data.normal_(0, 1.0 / float(n))
                m.conv.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, freq_num=64, reduction=16):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SpikePool(pool=nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                          step=self.step),
                SpikePool(pool=nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                          step=self.step),
                myBatchNorm3d(bn=BN(planes * block.expansion), step=self.step)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, step=self.step, freq_num=freq_num, reduction=reduction))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, step=self.step, freq_num=freq_num, reduction=reduction))

        return nn.Sequential(*layers)
    
    def forward(self, x, is_adain=False, is_drop=False, feat=None):

        if len(x.shape) == 4:
            x = x.repeat(self.step, 1, 1, 1, 1)
        else:
            x = x.permute([1, 0, 2, 3, 4])
        
        if not feat is None:
            x = self.fc(feat)
            return x
        else:
            x = self.conv1(x)
            x = self.avgpool(x)     # (T, B, 64, 16, 16)
            temp, x = self.layer1((x, x))   # (T, B, 64, 16, 16)
            temp, x = self.layer2((temp, x))    # (T, B, 128, 8, 8)
            temp, x = self.layer3((temp, x))    # (T, B, 256, 4, 4)
            temp, x = self.layer4((temp, x))    # (T, B, 512, 2, 2)

            if is_drop:
                temp = F.relu(temp)
                x = self.avgpool2(temp)
            else:
                x = self.avgpool2(x)
            
            if len(x.shape) == 4:
                x = x.view(x.size(0), x.size(1), -1)
                fea = x.mean([0])
            elif len(x.shape) == 5:
                x = x.view(x.size(0), x.size(1), -1)
                fea = x.mean([0])
            
            x = self.fc(x)

            if len(x.shape) == 3:
                x = x.mean([0])
            if is_adain:
                return fea, x
            else:
                return x

class ResNet(nn.Module):

    def __init__(self, block, layers, input_c=3, num_classes=1000, step=2, freq_num=49, reduction=1, pretrained_cfg=None, pretrained_cfg_overlay=None):
        super(ResNet, self).__init__()

        global BN
        BN = nn.BatchNorm2d
        global ReLU
        ReLU = nn.ReLU
        inplanes = 64
        self.inplanes = 64
        self.step = step

        self.conv1 = SpikePool(pool=nn.Conv2d(input_c, inplanes, kernel_size=7, stride=2, padding=3, bias=False), step=self.step)
        self.bn1 = myBatchNorm3d(bn=BN(self.inplanes), step=step)
        # self.relu = ReLU(inplace=True)
        self.relu = DctSpatialLIF(step=self.step, channel=64, h=32, w=32, freq_num=freq_num, reduction=reduction)
        # self.relu = LIFAct(step=self.step, channel=self.inplanes, dim=self.inplanes, update=update)
        self.avgpool1 = SpikePool(pool=nn.AvgPool2d(3,2,1), step=self.step)
        self.layer1 = self._make_layer(block, inplanes, layers[0], freq_num=freq_num, reduction=reduction)
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2, freq_num=freq_num, reduction=reduction)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2, freq_num=freq_num, reduction=reduction)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2, freq_num=freq_num, reduction=reduction)
        self.avgpool2 = SpikePool(pool=nn.AdaptiveAvgPool2d(1), step=self.step)
        self.fc = SpikeConv(conv=nn.Linear(inplanes * 8 * block.expansion, num_classes), step=self.step)

        for m in self.modules():
            if isinstance(m, SpikePool) and isinstance(m.pool, nn.Conv2d):
                n = m.pool.kernel_size[0] * m.pool.kernel_size[1] * m.pool.out_channels
                m.pool.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, myBatchNorm3d):
                m.bn.weight.data.fill_(1)
                m.bn.bias.data.zero_()
            elif isinstance(m, SpikeConv):
                n = m.conv.weight.size(1)
                m.conv.weight.data.normal_(0, 1.0 / float(n))
                m.conv.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, freq_num=49, reduction=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                SpikePool(pool=nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), step=self.step),
                myBatchNorm3d(bn=BN(planes * block.expansion), step=self.step)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, step=self.step, freq_num=freq_num, reduction=reduction))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, step=self.step, freq_num=freq_num, reduction=reduction))

        return nn.Sequential(*layers)

    def forward(self, x, is_adain=False, is_drop=False, feat=None):
        if len(x.shape) == 4:
            x = x.repeat(self.step, 1, 1, 1, 1)
        else:
            x = x.permute([1, 0, 2, 3, 4])

        if not feat is None:
            x = self.fc(feat)
            return x         
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.avgpool1(x)

            temp, x = self.layer1((x,x))
            temp, x = self.layer2((temp,x))
            temp, x = self.layer3((temp,x))
            temp, x = self.layer4((temp,x))
            
            if is_drop:
                x = self.avgpool2(temp)
            else:
                x = self.avgpool2(x)
                
            if len(x.shape) == 4:
                x = x.view(x.size(0), -1)
                fea = x
            elif len(x.shape) == 5:
                x = x.view(x.size(0), x.size(1), -1)
                fea = x.mean([0])
            x = self.fc(x)

            if len(x.shape) == 3:
                x = x.mean([0])
            if is_adain:
                return fea,x
            else:
                return x
            
@register_model
def DctResnet20_LIF(pretrained=False, **kwargs):
    model = ResNet20(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

@register_model
def DctResnet19_LIF(pretrained=False, **kwargs):
    model = ResNet19(BasicBlock, [3, 3, 2], **kwargs)
    return model

@register_model
def DctResNet18(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

@register_model
def DctResNet34(pretrained=False, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)