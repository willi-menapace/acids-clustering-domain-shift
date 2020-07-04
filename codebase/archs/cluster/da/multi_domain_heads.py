import torch
import torch.nn as nn

from codebase.archs.cluster.da.dial_resnet18_two_head import DialResNet18TwoHeadHead


class MultiDomainHeads(nn.Module):

    def __init__(self, config):
        super(MultiDomainHeads, self).__init__()

        self.config = config
        self.domains_count = config.domains_count
        self.num_sub_heads = config.num_sub_heads

        # Creates separate heads for each domain
        self.modules_map = nn.ModuleDict()
        for domain_idx in range(self.domains_count):
            for head in ('A', 'B'):
                outputs_count = self.config.output_k_B
                if head == 'A':
                    outputs_count = self.config.output_k_A

                self.modules_map[self.module_name_from_params(domain_idx, head)] = DialResNet18TwoHeadHead(self.config, outputs_count)

        self._initialize_weights()

    def module_name_from_params(selfdo, domain_idx, head):
        return str(domain_idx) + "_" + str(head)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                # nn.init.eye_(m.weight) # Not present in original code. Cannot use with multiple heads, otherwise there will be symmetries
                m.bias.data.zero_()

    def forward(self, x, domain_idx, head):
        '''

        :param x: Tensors to forward
        :param domain_idx: Domain id to which the tensors belong to
        :param head: character identifiying the head to which to forward the images
        :return:
        '''
        current_head = self.modules_map[self.module_name_from_params(domain_idx, head)]

        return current_head(x)