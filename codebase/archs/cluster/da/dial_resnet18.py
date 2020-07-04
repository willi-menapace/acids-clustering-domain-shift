import torch.nn as nn

from .dial import Dial2d, Dial1d, DialWhitening2d
from .dial_residual import DialBasicBlock, DialResNet, DialResNetTrunk, DialWhitenBasicBlock

# resnet18 and full channels

__all__ = ["DialResNet18"]

class DialResNet18Trunk(DialResNetTrunk):
    def __init__(self, config, logger=None):
        super(DialResNet18Trunk, self).__init__()

        # If present register the logger
        self.logger = logger
        self.config = config

        self.batchnorm_track = config.batchnorm_track

        block = DialBasicBlock
        whiten_block = DialWhitenBasicBlock
        layers = [2, 2, 2, 2]

        in_channels = config.in_channels
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1,
                               padding=1,
                               bias=False)
        if config.whiten:
            self.bn1 = DialWhitening2d(64, config, config.all_domains_count, track_running_stats=self.batchnorm_track)
        else:
            self.bn1 = Dial2d(64, config, config.all_domains_count, track_running_stats=self.batchnorm_track)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        if config.whiten:
            self.layer1 = self._make_layer(whiten_block, DialWhitening2d, 64, layers[0])
        else:
            self.layer1 = self._make_layer(block, Dial2d, 64, layers[0])
        self.layer2 = self._make_layer(block, Dial2d, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, Dial2d, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, Dial2d, 512, layers[3], stride=2)

        if config.input_sz == 128:
            avg_pool_sz = 9
        if config.input_sz == 96:
            avg_pool_sz = 7
        elif config.input_sz == 64:
            avg_pool_sz = 5
        elif config.input_sz == 32:
            avg_pool_sz = 3
        print("avg_pool_sz %d" % avg_pool_sz)

        self.avgpool = nn.AvgPool2d(avg_pool_sz, stride=1)

    def forward(self, x, domains, is_transformed, penultimate_features=False, ground_truth=None):
        x = self.conv1(x)
        x = self.bn1(x, domains, is_transformed)
        x = self.relu(x)

        if self.logger:
            self.logger.add_tensor("layer_0", x.mean(3).mean(2), ground_truth)

        x = self.maxpool(x)

        x = self.layer1(x, domains, is_transformed)
        if self.logger:
            self.logger.add_tensor("layer_1", x.mean(3).mean(2), ground_truth)
        x = self.layer2(x, domains, is_transformed)
        if self.logger:
            self.logger.add_tensor("layer_2", x.mean(3).mean(2), ground_truth)
        x = self.layer3(x, domains, is_transformed)
        if self.logger:
            self.logger.add_tensor("layer_3", x.mean(3).mean(2), ground_truth)

        unaveraged_features = None
        if not penultimate_features:
            # default
            x = self.layer4(x, domains, is_transformed)
            if self.logger:
                self.logger.add_tensor("layer_4", x.mean(3).mean(2), ground_truth)

            unaveraged_features = x
            x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        return x, unaveraged_features


class DialResNet18Head(nn.Module):
    def __init__(self, config):
        super(DialResNet18Head, self).__init__()

        self.batchnorm_track = config.batchnorm_track

        self.num_sub_heads = config.num_sub_heads

        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(512 * DialBasicBlock.expansion, config.output_k),
            nn.Softmax(dim=1)) for _ in range(self.num_sub_heads)])

    def forward(self, x, kmeans_use_features=False):
        results = []
        for i in range(self.num_sub_heads):
            if kmeans_use_features:
                results.append(x)  # duplicates
            else:
                results.append(self.heads[i](x))
        return results


class DialResNet18(DialResNet):
    def __init__(self, config):
        # no saving of configs
        super(DialResNet18, self).__init__()

        self.batchnorm_track = config.batchnorm_track

        self.trunk = DialResNet18Trunk(config)
        self.head = DialResNet18Head(config)

        self._initialize_weights()

    def forward(self, x, domains, is_transformed, kmeans_use_features=False, trunk_features=False,
                penultimate_features=False):
        x, unaveraged_features = self.trunk(x, domains, is_transformed, penultimate_features=penultimate_features)

        if trunk_features:  # for semisup
            return x

        x = self.head(x, kmeans_use_features=kmeans_use_features)  # returns list
        return x
