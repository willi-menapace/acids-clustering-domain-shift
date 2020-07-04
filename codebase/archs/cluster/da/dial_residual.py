import torch.nn as nn

from .dial import Dial2d, Dial1d, DialWhitening2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class DialBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, config, stride=1, downsample=None, domains_count=None, track_running_stats=None):
        super(DialBasicBlock, self).__init__()

        assert(domains_count is not None)
        assert(track_running_stats is not None)

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = Dial2d(planes, config, domains_count, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = Dial2d(planes, config, domains_count, track_running_stats=track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, domains, is_transformed):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, domains, is_transformed)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, domains, is_transformed)

        if self.downsample is not None:
            residual = self.downsample(x, domains, is_transformed)

        out += residual
        out = self.relu(out)

        return out

class DialWhitenBasicBlock(DialBasicBlock):
    def __init__(self, inplanes, planes, config, stride=1, downsample=None, domains_count=None, track_running_stats=None):
        super(DialWhitenBasicBlock, self).__init__(inplanes, planes, config, stride, downsample, domains_count, track_running_stats)

        if not self.downsample is None:
            raise Exception("Downsample layers may contain non whitened BN, please implement conversion is you which to use downsampling with whitening")

        self.bn1 = DialWhitening2d(planes, config, domains_count, track_running_stats=track_running_stats)
        self.bn2 = DialWhitening2d(planes, config, domains_count, track_running_stats=track_running_stats)

class DialSequential(nn.Sequential):
    '''
    A Dial-aware sequential layer that allows feeding domain information through a series of mixed Dial and non-Dial layers
    '''
    def forward(self, input, domains, is_transformed):
        '''
        Forwards the output of each module as input to the successive. Each dial layer is also fed with the domains input
        :param input: Initial input to feed to the first layer
        :param domains: Input to feed to every dial layer
        :return: the output of the last layer
        '''
        for module in self._modules.values():
            if isinstance(module, Dial2d) or isinstance(module, Dial1d) or isinstance(module, DialSequential) or isinstance(module, DialBasicBlock):
                input = module(input, domains, is_transformed)
            else:
                input = module(input)
        return input

class DialResNetTrunk(nn.Module):
    def __init__(self):
        super(DialResNetTrunk, self).__init__()

    def _make_layer(self, block, bn_block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DialSequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                bn_block(planes * block.expansion, self.config, self.config.all_domains_count, track_running_stats=self.batchnorm_track),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.config, stride, downsample, domains_count=self.config.all_domains_count,
                            track_running_stats=self.batchnorm_track))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, self.config, domains_count=self.config.all_domains_count, track_running_stats=self.batchnorm_track))

        return DialSequential(*layers)


class DialResNet(nn.Module):
    def __init__(self):
        super(DialResNet, self).__init__()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, Dial2d):
                #Do nothing because initialization is performed by the block itself
                pass
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                #nn.init.eye_(m.weight) # Not present in original code. Cannot use with multiple heads, otherwise there will be symmetries
                m.bias.data.zero_()
