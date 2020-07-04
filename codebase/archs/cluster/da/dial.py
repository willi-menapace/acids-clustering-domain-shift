import torch
import torch.nn as nn

from codebase.archs.cluster.da.whitening import Whitening2d


class DialBase(nn.Module):
    def __init__(self, config):
        super(DialBase, self).__init__()

        self.config = config

    def create_domain_ranges(self, input, domains):
        '''
        Divides the inputs in ranges with the same domain

        :param input: see forward
        :param domains: see forward
        :return: list of (inputs, domain_id) representing inputs belonging all to the same domain id
        '''

        results = []

        # If domains is a scalar all items are from the same domain
        if domains.size()[0] == 1:
            return [(input, domains.item())]

        batch_size = domains.size()[0]
        current_domain_id = domains[0, 0]
        current_domain_start = 0
        for i in range(1, batch_size):
            latest_domain = domains[i, 0]
            if latest_domain != current_domain_id:
                results.append([input[current_domain_start:i], current_domain_id.item()])
                current_domain_id = latest_domain
                current_domain_start = i
        # Appends last domain
        results.append([input[current_domain_start:batch_size], current_domain_id.item()])

        return results

    def forward_domain(self, input, domain_id, is_transformed):
        '''

        :param input: (bs, channels, rows, cols) tensor with inputs belonging all to the same domain
        :param domain_id: python integer representing the domain id
        :param is_transformed: True if the batch corresponds to transformed images, False otherwise
        :return: batch normalized (bs, channels, rows, cols) tensor
        '''

        mapped_domain = self.config.domain_permutation[domain_id] # User may ask to use a different mapping between domain id and bn layers
        if is_transformed and self.config.transformed_is_domain: # The second half of domains correspond to transformed domains
            mapped_domain += self.domains_count
        output = self.batch_norm_layers[mapped_domain](input)

        # If we need common gamma and beta parameters for all domains
        if not self.config.domain_specific_bn_affine:
            output = output * self.gamma + self.beta
        return output

    def forward(self, input, domains, is_transformed):
        '''

        :param input: (bs, ...) tensor
        :param domains: (bs, 1) tensor with domain labels. All labels relative to the same domain must be compact. Alternatively
                        accepts a scalar tensor representing the domain of all inputs
        :param is_transformed: True if the batch corresponds to transformed images, False otherwise
        :return: batch normalized (bs, ...) tensor with preserved input order
        '''

        # Separates inputs into domains
        domain_ranges = self.create_domain_ranges(input, domains)

        results = []

        # Forwards all domains
        for current_input, current_domain in domain_ranges:
            current_result = self.forward_domain(current_input, current_domain, is_transformed)
            results.append(current_result)

        # Concatenates all results and returns them
        # so that output results order is the same as the input order
        return torch.cat(results, dim=0)

class Dial2d(DialBase):
    '''
    Dial BN for 2d convolutions

    Supports (bs, channels, rows, columns) tensors
    '''
    def __init__(self, channels, config, domains_count, track_running_stats=True):
        '''
        Creates a 2d Dial layer supporting domain ids [0, domains_count-1]
        :param domains_count: number of domains
        :param channels: number of channels
        '''
        super(Dial2d, self).__init__(config)

        self.channels = channels
        self.domains_count = domains_count

        # Learnable common scaling parameters. Used only then not using domain specific bn affine
        self.gamma = nn.Parameter(torch.ones(channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(channels, 1, 1))

        self.batch_norm_layers = nn.ModuleList()

        total_domains = domains_count
        if config.transformed_is_domain:
            total_domains *= 2 # create a domain also for augmented_images
        for i in range(total_domains):
            self.batch_norm_layers.append(nn.BatchNorm2d(channels, affine=self.config.domain_specific_bn_affine, track_running_stats=track_running_stats))

class Dial1d(DialBase):
    '''
    Dial BN for 2d convolutions

    Supports (bs, channels, rows, columns) tensors
    '''
    def __init__(self, outputs, config, domains_count, track_running_stats=True):
        '''
        Creates a 2d Dial layer supporting domain ids [0, domains_count-1]
        :param domains_count: number of domains
        :param channels: number of channels
        '''
        super(Dial1d, self).__init__(config)

        self.outputs = outputs
        self.domains_count = domains_count

        # Learnable common scaling parameters. Used only then not using domain specific bn affine
        self.gamma = nn.Parameter(torch.ones(1, outputs))
        self.beta = nn.Parameter(torch.zeros(1, outputs))

        self.batch_norm_layers = nn.ModuleList()
        total_domains = domains_count
        if config.transformed_is_domain:
            total_domains *= 2  # create a domain also for augmented_images
        for i in range(total_domains):
            self.batch_norm_layers.append(nn.BatchNorm1d(outputs, affine=self.config.domain_specific_bn_affine, track_running_stats=track_running_stats))


class DialWhitening2d(DialBase):
    '''
    Dial Whitening for 2d convolutions

    Supports (bs, channels, rows, columns) tensors
    '''
    def __init__(self, channels, config, domains_count, track_running_stats=True):
        '''
        Creates a 2d Dial layer supporting domain ids [0, domains_count-1]
        :param domains_count: number of domains
        :param channels: number of channels
        '''
        super(DialWhitening2d, self).__init__(config)

        self.channels = channels
        self.domains_count = domains_count

        # Learnable common scaling parameters. Used only then not using domain specific bn affine
        self.gamma = nn.Parameter(torch.ones(channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(channels, 1, 1))

        self.batch_norm_layers = nn.ModuleList()

        total_domains = domains_count
        if config.transformed_is_domain:
            total_domains *= 2 # create a domain also for augmented_images
        for i in range(total_domains):
            self.batch_norm_layers.append(Whitening2d(channels, affine=self.config.domain_specific_bn_affine, track_running_stats=track_running_stats))

if __name__ == "__main__":

    batch_size = 100
    channels = 128
    outputs = 1024
    dimensions = 10

    test_1dinput = torch.zeros((batch_size, outputs), dtype=torch.float32)
    test_2dinput = torch.zeros((batch_size, channels, dimensions, dimensions), dtype=torch.float32)
    test_domains = torch.tensor([0] * (batch_size // 4) + [1] * (batch_size // 4) + [2] * (batch_size // 4) + [3] * (batch_size // 4), dtype=torch.int32)
    test_domains = test_domains.reshape((batch_size, 1))

    dial2d = Dial2d(channels, 4)
    output = dial2d(test_2dinput, test_domains)

    #print(output)

    dial1d = Dial1d(outputs, 4)
    output = dial1d(test_1dinput, test_domains)

    print(output)
