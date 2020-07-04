import torch
import torch.nn as nn

from .dial_resnet18 import DialResNet18Trunk
from .dial_residual import DialBasicBlock, DialResNet

from tensorlog.tensor_logger import TensorLogger

__all__ = ["DialResNet18TwoHead"]

class DialResNet18TwoHeadHead(nn.Module):
    def __init__(self, config, output_k, semisup=False):
        super(DialResNet18TwoHeadHead, self).__init__()

        self.batchnorm_track = config.batchnorm_track

        self.semisup = semisup

        self.output_k = output_k
        # Statistics about logits
        self.logit_variance_sum = torch.zeros((output_k), dtype=torch.float).cuda()
        self.logit_sum_count = 0

        if not semisup:
            self.num_sub_heads = config.num_sub_heads

            self.heads = nn.ModuleList([nn.Sequential(
                nn.Linear(512 * DialBasicBlock.expansion, output_k),
                nn.Softmax(dim=1)) for _ in range(self.num_sub_heads)])
        else:
            self.head = nn.Linear(512 * DialBasicBlock.expansion, output_k)

    def forward(self, x, kmeans_use_features=False):
        if not self.semisup:
            results = []
            results_logit = []

            for i in range(self.num_sub_heads):
                if kmeans_use_features:
                    results.append(x)  # duplicates
                else:
                    modules = list(self.heads[i].modules())
                    # module 0 is the sequential layer
                    logit_output = modules[1](x)

                    softmax_output = modules[2](logit_output)
                    results.append(softmax_output)
                    results_logit.append(logit_output)

            return results, results_logit
        else:

            return self.head(x)


class DialResNet18TwoHead(DialResNet):
    def __init__(self, config):
        # no saving of configs
        super(DialResNet18TwoHead, self).__init__()

        self.logger = TensorLogger()
        self.logger.enabled = False

        self.batchnorm_track = config.batchnorm_track

        self.trunk = DialResNet18Trunk(config, self.logger)

        self.dropout = nn.Dropout(config.dropout_probability)

        self.head_A = DialResNet18TwoHeadHead(config, output_k=config.output_k_A)

        semisup = (hasattr(config, "semisup") and config.semisup)
        print("semisup: %s" % semisup)

        self.head_B = DialResNet18TwoHeadHead(config, output_k=config.output_k_B, semisup=semisup)

        self._initialize_weights()

    def get_and_reset_logit_variances(self):
        '''
        Returns the average variance of logits over the mean of all class outputs
        Resets internal counters for logit variances
        :return: tuple (logit_variance_head_A, logit_variance_head_b)
        '''
        return self.head_A.get_and_reset_logit_variance(), self.head_B.get_and_reset_logit_variance()

    def forward(self, x, domains, is_transformed, head="B", kmeans_use_features=False, ground_truth=None):
        # default is "B" for use by eval codebase
        # training script switches between A and B

        tf, unaveraged_features = self.trunk(x, domains, is_transformed, ground_truth=ground_truth)

        tf_dropout = self.dropout(tf)

        # returns list or single
        if head == "A":
            x, logits = self.head_A(tf_dropout, kmeans_use_features=kmeans_use_features)
        elif head == "B":
            x, logits = self.head_B(tf_dropout, kmeans_use_features=kmeans_use_features)
        else:
            assert (False)

        self.logger.add_tensor("head_0", x[0], ground_truth)

        return x, logits, tf, unaveraged_features

