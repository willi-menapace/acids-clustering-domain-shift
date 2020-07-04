import sys

import torch

import codebase.archs as archs
from codebase.archs.cluster.estimator import FixedMatrixEstimator

class BasicDialNetworkManager:

    def __init__(self, config):
        self.config = config
        self.domains_count = config.domains_count

        # Creates domain adaptive encoder network
        self.encoder_network = archs.da.__dict__[config.encoder_arch](config)

        # Creates the estimators for the joint probability.
        # Each domain has two estimators for the two heads
        self.estimators = []
        for current_domain_idx in range(self.domains_count):
            self.estimators.append({'A': FixedMatrixEstimator(config.output_k_A, config.output_k_A, config.joint_alpha), 'B': FixedMatrixEstimator(config.output_k_B, config.output_k_B, config.joint_alpha)})

    def compute_joint(self, plain_cluster_probabilities, transformed_cluster_probabilities):
        '''
        Computes the joint probability matrix
        :param plain_cluster_probabilities: (batch_size, cluster number) tensor of cluster assignment probabilities for the plain images
        :param transformed_cluster_probabilities: (batch_size, cluster number) tensor of cluster assignemnt probabilities for the augmented images
        :return: (cluster, cluster number) tensor joint probability matrix
        '''

        bn, k = plain_cluster_probabilities.size()
        assert (transformed_cluster_probabilities.size(0) == bn and transformed_cluster_probabilities.size(1) == k)

        p_i_j = plain_cluster_probabilities.unsqueeze(2) * transformed_cluster_probabilities.unsqueeze(1)  # bn, k, k
        p_i_j = p_i_j.sum(dim=0)  # k, k
        p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
        p_i_j = p_i_j / p_i_j.sum()  # normalise

        return p_i_j

    def main_cluster_loss(self, joint_probability_matrix, EPS=sys.float_info.epsilon):
        '''
        Computes the main clusterin loss starting from the estimated joint probability matrix
        :param joint_probability_matrix:  estimated probability matrix of cluster assignment of plain vs transformed images
        :param lamb:
        :param EPS:
        :return: the main clusterint loss
        '''
        k = joint_probability_matrix.size()[0]

        assert (joint_probability_matrix.size() == (k, k))

        marginal_i = joint_probability_matrix.sum(dim=1).view(k, 1).expand(k, k)
        marginal_j = joint_probability_matrix.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

        # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
        joint_probability_matrix[(joint_probability_matrix < EPS).data] = EPS
        marginal_j[(marginal_j < EPS).data] = EPS
        marginal_i[(marginal_i < EPS).data] = EPS

        loss = - joint_probability_matrix * (torch.log(joint_probability_matrix) \
                                             - self.config.lamb * torch.log(marginal_j) \
                                             - self.config.lamb * torch.log(marginal_i))

        loss = loss.sum()

        return loss

    def check_invalid_values(self, tensor):
        if (torch.isnan(tensor) | torch.isnan(tensor)).any():
            raise Exception("Invalid values found")
        if (torch.isinf(tensor) | torch.isinf(tensor)).any():
            raise Exception("Invalid values found")

    def compute_cluster_losses(self, training_batch, domain_id, head):
        '''
        Performs computation of the loss terms associated to clustering for the specified domain.
        :param training_batch: BasicDialTrainBatch data for the current step
        :return:
        '''

        # Get current batches in the form (images_tensors, domain_id)
        domain_batch_plain, domain_batch_transformed = training_batch.get_batches_by_domain(domain_id)

        self.check_invalid_values(domain_batch_plain[0])
        self.check_invalid_values(domain_batch_transformed[0])

        plain_cluster_probabilities, plain_logits, plain_features = self.encoder_network(domain_batch_plain[0], domain_batch_plain[1], head=head)
        transformed_cluster_probabilities, transformed_logits, transformed_features = self.encoder_network(domain_batch_transformed[0], domain_batch_transformed[1], head=head)

        assert(len(plain_cluster_probabilities) == 1) # Support only one sub head
        plain_cluster_probabilities = plain_cluster_probabilities[0]
        plain_logits = plain_logits[0]
        assert (len(transformed_cluster_probabilities) == 1)  # Support only one sub head
        transformed_cluster_probabilities = transformed_cluster_probabilities[0]
        transformed_logits = transformed_logits[0]

        current_estimator = self.estimators[domain_id][head]

        current_joint_probability = self.compute_joint(plain_cluster_probabilities, transformed_cluster_probabilities)
        estimated_joint_probability = current_estimator(current_joint_probability)

        self.check_invalid_values(estimated_joint_probability)

        return self.main_cluster_loss(estimated_joint_probability)

    def compute_supervised_losses(self, training_batch, domain_id):
        '''
        Computes the Cross entropy loss for the given training batch and input domain
        :param training_batch:
        :param domain_id:
        :return:
        '''
        # Get current batches in the form (images_tensors, domain_id)
        _, domain_batch_transformed = training_batch.get_batches_by_domain(domain_id)
        images_tensor = domain_batch_transformed[0]
        domain_tensor = domain_batch_transformed[1]
        ground_truth_tensor = domain_batch_transformed[2]

        loss = torch.nn.CrossEntropyLoss()
        _, transformed_logits, _ = self.encoder_network(domain_batch_transformed[0], domain_batch_transformed[1], head='B')
        return loss(transformed_logits[0], ground_truth_tensor.type(torch.LongTensor).cuda())

    def load(self, state_dict):
        '''
        Loads state from dictionary. The dictionary will contain all keys returned by calls to get_state

        :param state_dict:
        :return:
        '''
        self.encoder_network.load_state_dict(state_dict["encoder"])

        # Loads the estimators state
        for current_domain_idx in range(self.domains_count):
            state_dict_a = state_dict["estimator_A_domain_{}".format(current_domain_idx)]
            state_dict_b = state_dict["estimator_B_domain_{}".format(current_domain_idx)]

            self.estimators[current_domain_idx]['A'].load_state_dict(state_dict_a)
            self.estimators[current_domain_idx]['B'].load_state_dict(state_dict_b)

    def get_state(self):
        '''
        Gets the state dictionary representing the current state of the network
        :return:
        '''
        state_dict = {}

        state_dict["encoder"] = self.encoder_network.state_dict()

        # Saves the estimators state
        for current_domain_idx in range(self.domains_count):
            state_dict["estimator_A_domain_{}".format(current_domain_idx)] = self.estimators[current_domain_idx]['A'].state_dict()
            state_dict["estimator_B_domain_{}".format(current_domain_idx)] = self.estimators[current_domain_idx]['B'].state_dict()

        return state_dict

    def train(self):
        self.encoder_network.train()

    def eval(self):
        self.encoder_network.eval()

    def to_parallel_gpu(self):
        '''
        Transforms all network elements to cuda and where possible splits them across gpus
        :return:
        '''
        self.encoder_network.cuda()
        self.encoder_network = torch.nn.DataParallel(self.encoder_network)

        for current_estimator_dict in self.estimators:
            current_estimator_dict['A'].cuda()
            current_estimator_dict['B'].cuda()
