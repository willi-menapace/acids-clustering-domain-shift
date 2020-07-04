import sys

import torch

import codebase.archs as archs
from codebase.archs.cluster.da.multi_domain_heads import MultiDomainHeads

class RefinementNetworkManager:

    def __init__(self, config):
        self.config = config
        self.domains_count = config.domains_count
        self.num_sub_heads = config.num_sub_heads
        self.encoder_loaded = False

        # Creates domain adaptive encoder network
        self.encoder_network = archs.da.__dict__[config.encoder_arch](config)

        self.encoder_network.eval() # The encoder is not trained

        self.heads = MultiDomainHeads(config)

    def load_encoder_state(self, encoder_state):
        self.encoder_loaded = True
        self.encoder_network.load_state_dict(encoder_state)  # Load pretrained weights

    def compute_cluster_joint(self, plain_cluster_probabilities, transformed_cluster_probabilities):
        '''
        Computes the joint probability matrix
        :param plain_cluster_probabilities: (batch_size, cluster number) tensor of cluster assignment probabilities for the plain images
        :param transformed_cluster_probabilities: (batch_size, cluster number) tensor of cluster assignemnt probabilities for the augmented images
        :return: (cluster, cluster number) tensor joint probability matrix
        '''

        if not self.encoder_loaded:
            raise Exception("Attempting to use uninitialized Refinement network. Please load encoder state.")

        bn, k = plain_cluster_probabilities.size()
        assert (transformed_cluster_probabilities.size(0) == bn and transformed_cluster_probabilities.size(1) == k)

        p_i_j = plain_cluster_probabilities.unsqueeze(2) * transformed_cluster_probabilities.unsqueeze(1)  # bn, k, k
        p_i_j = p_i_j.sum(dim=0)  # k, k
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise

        return p_i_j

    def mutual_information(self, joint_probability_matrix, EPS=sys.float_info.epsilon):
        '''
        Computes the mutual information of the joint probability matrix
        :param joint_probability_matrix:  probability matrix tensor for which to compute mutual information
        :param EPS:
        :return: the mutual information
        '''

        rows, columns = joint_probability_matrix.size()

        marginal_rows = joint_probability_matrix.sum(dim=1).view(rows, 1).expand(rows, columns)
        marginal_columns = joint_probability_matrix.sum(dim=0).view(1, columns).expand(rows, columns)  # but should be same, symmetric

        # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
        joint_probability_matrix[(joint_probability_matrix < EPS).data] = EPS
        marginal_rows[(marginal_rows < EPS).data] = EPS
        marginal_columns[(marginal_columns < EPS).data] = EPS

        mutual_information = joint_probability_matrix * (torch.log(joint_probability_matrix) \
                                             - self.config.lamb * torch.log(marginal_rows) \
                                             - self.config.lamb * torch.log(marginal_columns))

        mutual_information = mutual_information.sum()

        return mutual_information

    def create_domain_ranges(self, input, domains):
        '''
        Divides the inputs in ranges with the same domain

        :param input: (bs, ...) tensor
        :param domains: (bs, 1) tensor with domain labels. All labels relative to the same domain must be compact. Alternatively
                        accepts a scalar tensor representing the domain of all inputs
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

    def compute_losses(self, training_batch, head):
        '''
        Performs computation of the loss terms associated to clustering
        :param training_batch: MixedDomainTrainBatch data for the current step
        :return:
        '''

        # Get current batches in the form (images_tensors, domain_id, ground_truth)
        domain_batch_plain, domain_batch_transformed = training_batch.get_batches()

        # Encoder computations do not require gradient and backpropagation
        with torch.no_grad():
            _, _, plain_features, _ = self.encoder_network(domain_batch_plain[0], domain_batch_plain[1], head=head)
            _, _, transformed_features, _ = self.encoder_network(domain_batch_transformed[0], domain_batch_transformed[1], head=head)

        # No need for gradient propagation inside encoder network
        plain_features = plain_features.detach()
        transformed_features = transformed_features.detach()

        total_mi_loss = torch.tensor([0.0]).cuda()

        per_domain_plain_features = self.create_domain_ranges(plain_features, domain_batch_plain[1])
        per_domain_transformed_features = self.create_domain_ranges(transformed_features, domain_batch_transformed[1])

        assert (len(per_domain_plain_features) == self.domains_count)
        assert (len(per_domain_transformed_features) == self.domains_count)

        for domain_idx, (current_domain_plain, current_domain_transformed) in enumerate(zip(per_domain_plain_features, per_domain_transformed_features)):
            # Check features correspond to the same domain
            assert (current_domain_plain[1] == current_domain_transformed[1])
            # Checks ids are ordered
            assert (domain_idx == current_domain_plain[1])

            current_domain_plain_probabilities, _ = self.heads(current_domain_plain[0], domain_idx, head)
            current_domain_transformed_probabilities, _ = self.heads(current_domain_transformed[0], domain_idx, head)

            # Computes losses for each sub head
            for current_subhead_index in range(self.num_sub_heads):
                current_subhead_domain_plain_probabilities = current_domain_plain_probabilities[current_subhead_index]
                current_subhead_domain_transformed_probabilities = current_domain_transformed_probabilities[current_subhead_index]

                joint_probability = self.compute_cluster_joint(current_subhead_domain_plain_probabilities, current_subhead_domain_transformed_probabilities)
                total_mi_loss += -self.mutual_information(joint_probability) # Everything is accumulated in the same loss and is rescaled only at the end

        # Computes average losses over all subheads
        # total loss is not divided by the number of domain to avoid gradients rescaling
        return total_mi_loss / self.num_sub_heads

    def load(self, state_dict):
        '''
        Loads state from dictionary. The dictionary will contain all keys returned by calls to get_state

        :param state_dict:
        :return:
        '''
        self.heads.load_state_dict(state_dict["heads"])

    def get_state(self):
        '''
        Gets the state dictionary representing the current state of the network
        :return:
        '''
        state_dict = {}

        state_dict["heads"] = self.heads.state_dict()

        return state_dict

    def train(self):
        self.heads.train()

    def eval(self):
        self.heads.eval()

    def to_parallel_gpu(self):
        '''
        Transforms all network elements to cuda and where possible splits them across gpus
        :return:
        '''
        self.encoder_network.cuda()
        self.encoder_network = torch.nn.DataParallel(self.encoder_network)

        self.heads.cuda()

