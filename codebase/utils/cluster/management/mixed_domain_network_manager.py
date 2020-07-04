import sys

import torch

import codebase.archs as archs
from codebase.archs.cluster.estimator import FixedMatrixEstimator, CentroidEstimator


class MixedDomainNetworkManager:

    def __init__(self, config):
        self.config = config
        self.domains_count = config.domains_count
        self.test_domains_count = config.test_domains_count
        self.num_sub_heads = config.num_sub_heads

        # Creates domain adaptive encoder network
        self.encoder_network = archs.da.__dict__[config.encoder_arch](config)

        # Creates the estimators for the joint probability.
        self.main_estimators_map = {'A': [FixedMatrixEstimator(config.output_k_A, config.output_k_A, config.joint_alpha) for _ in range(self.num_sub_heads)], 'B': [FixedMatrixEstimator(config.output_k_B, config.output_k_B, config.joint_alpha) for _ in range(self.num_sub_heads)]}

        # Creates the estimators for the domain-cluster joint probability
        self.domains_count_multiplier = 1
        # If augmented images have their own domains we need double the number of domains
        if self.config.transformed_is_domain:
            self.domains_count_multiplier = 2
        self.domain_estimators_map = {'A': [FixedMatrixEstimator(self.domains_count * self.domains_count_multiplier, config.output_k_A, config.joint_alpha) for _ in range(self.num_sub_heads)], 'B': [FixedMatrixEstimator(self.domains_count * self.domains_count_multiplier, config.output_k_B, config.joint_alpha) for _ in range(self.num_sub_heads)]}

        if self.config.domain_cluster_separate_transform:
            if self.config.transformed_is_domain:
                raise Exception("Transformed images are already separated, cannot use separate comdain cluster loss.")
        self.domain_estimators_transformed_map = {'A': [FixedMatrixEstimator(self.domains_count * self.domains_count_multiplier, config.output_k_A, config.joint_alpha) for _ in range(self.num_sub_heads)], 'B': [FixedMatrixEstimator(self.domains_count * self.domains_count_multiplier, config.output_k_B, config.joint_alpha) for _ in range(self.num_sub_heads)]}

        # Creates estimator for domain specific losses
        self.domain_specific_estimators_map = {}
        for domain_id in range(self.domains_count):
            self.domain_specific_estimators_map[('A', domain_id)] = [FixedMatrixEstimator(config.output_k_A, config.output_k_A, config.joint_alpha) for _ in range(self.num_sub_heads)]
            self.domain_specific_estimators_map[('B', domain_id)] = [FixedMatrixEstimator(config.output_k_B, config.output_k_B, config.joint_alpha) for _ in range(self.num_sub_heads)]

        self.centroid_estimators_map = {'A': [CentroidEstimator(config, config.output_k_A) for _ in range(self.num_sub_heads)], 'B': [CentroidEstimator(config, config.output_k_B) for _ in range(self.num_sub_heads)]}

    def compute_domain_cluster_joint(self, domain_assignments, cluster_probabilities):
        '''
        Computes the joint probability matrix between domain and cluster assignments
        :param domain_assignments: (batch_size, 1) integer tensor representing the domain of each sample
        :param cluster_probabilities: (batch_size, cluster number) tensor representing cluster assignemnt probabilities
        :return: (domain count, cluster number) tensor joint probability matrix
        '''

        batch_size_domains = domain_assignments.size()[0]
        batch_size, cluster_number = cluster_probabilities.size()

        assert(batch_size_domains == batch_size)

        domain_probabilities = torch.cuda.FloatTensor(batch_size, self.domains_count * self.domains_count_multiplier).zero_()
        domain_probabilities.scatter_(1, domain_assignments.type(torch.LongTensor).cuda(), 1)

        # Computes and normalizes the joint matrix
        joint_matrix = torch.t(domain_probabilities).matmul(cluster_probabilities)
        joint_matrix = joint_matrix / joint_matrix.sum()

        return joint_matrix

    def compute_cluster_joint(self, plain_cluster_probabilities, transformed_cluster_probabilities):
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

    def mutual_information(self, joint_probability_matrix, lamb, EPS=sys.float_info.epsilon):
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
                                             - lamb * torch.log(marginal_rows) \
                                             - lamb * torch.log(marginal_columns))

        mutual_information = mutual_information.sum()

        return mutual_information

    def compute_logit_variance_loss(self, logits):
        '''
        Computes loss penalizing the variance of the logits with respect to the mean across all output units
        :param logits: tensor (batch_size, output_units)
        :return: logit variance loss
        '''

        demeaned_logits = (logits - logits.mean(dim=1, keepdim=True)) # Subtracts the mean logit value across all output units
        demeaned_logits_variance = torch.pow(demeaned_logits, 2).mean(dim=0) # Computes variance for each output unit
        demeaned_logits_variance = demeaned_logits_variance.mean() # Computes average veriance across all output units

        return demeaned_logits_variance

    def check_invalid_values(self, tensor):
        if (torch.isnan(tensor) | torch.isnan(tensor)).any():
            raise Exception("Invalid values found")
        if (torch.isinf(tensor) | torch.isinf(tensor)).any():
            raise Exception("Invalid values found")

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

    def compute_centroid_loss(self, global_centroid : torch.Tensor, domain_centroids : torch.Tensor):

        num_domains = len(domain_centroids)
        num_clusters = global_centroid.size()[1]

        loss = torch.tensor([0.0]).cuda()
        for domain_index in range(num_domains):
            loss += torch.sum(torch.pow(global_centroid - domain_centroids[domain_index], 2))
        loss = loss / (num_domains * num_clusters)

        return loss

    def fixmatch_transformation(self, probabilities : torch.Tensor, threshold):
        '''
        Brings all probabilities above threshold to 1
        :param probabilities: (bs, num_classes) tensor representing class probabilites over the batch
        :param threshold: threshold value for row selection
        :return: (bs, num_classes) tensor with modified probabilities
        '''
        values, positions = torch.max(probabilities, dim=1)
        for idx, value in enumerate(values):
            # If the maximum prediction is above threshold, raise the prediction to 1
            # and put 0 anywhere else
            if value >= threshold:
                probabilities[idx] = 0.0
                probabilities[idx, positions[idx]] = 1.0

        return probabilities

    def compute_refinement_losses(self, training_batch, head, use_fixmatch=False, fixmatch_threshold=0.95):
        '''
        Performs computation of the IIC loss terms associated to refining for the test domain.
        :param training_batch: RefinementTrainBatch data for the current step
        :param use_fixmatch: If True uses fixmatch for the refinement, otherwise uses plain mutual information
        :param fixmatch_threshold: optional value indicating the threshold for fixmatch
        :return:
        '''
        # Get current batches in the form (images_tensors, domain_id, ground_truth)
        domain_batch_plain, domain_batch_transformed = training_batch.get_batches()

        all_plain_cluster_probabilities, all_plain_logits, plain_features, _ = self.encoder_network(domain_batch_plain[0], domain_batch_plain[1], is_transformed=False, head=head)
        all_transformed_cluster_probabilities, all_transformed_logits, transformed_features, _ = self.encoder_network(domain_batch_transformed[0], domain_batch_transformed[1], is_transformed=True, head=head)

        total_mi_loss = torch.tensor([0.0]).cuda()
        total_logit_loss = torch.tensor([0.0]).cuda()
        # Computes losses for each sub head
        for current_subhead_index in range(self.num_sub_heads):
            plain_cluster_probabilities = all_plain_cluster_probabilities[current_subhead_index]
            plain_logits = all_plain_logits[current_subhead_index]
            transformed_cluster_probabilities = all_transformed_cluster_probabilities[current_subhead_index]
            transformed_logits = all_transformed_logits[current_subhead_index]

            if use_fixmatch:
                plain_cluster_probabilities = self.fixmatch_transformation(plain_cluster_probabilities.clone(), fixmatch_threshold)



            current_main_estimator = self.main_estimators_map[head][current_subhead_index]

            # Estimates transformed images probability
            clustering_joint_probability = self.compute_cluster_joint(plain_cluster_probabilities,
                                                                      transformed_cluster_probabilities)
            estimated_clustering_joint_probability = current_main_estimator(clustering_joint_probability)

            self.check_invalid_values(estimated_clustering_joint_probability)

            all_logits = torch.cat([plain_logits, transformed_logits])
            logit_loss = self.compute_logit_variance_loss(all_logits)

            # Maximize mutual information between transformed images, minimize mutual information between cluster assignments and domain information
            total_mi_loss += -self.mutual_information(estimated_clustering_joint_probability, self.config.lamb)
            total_logit_loss += logit_loss

        # Computes average losses over all subheads
        return total_mi_loss / self.num_sub_heads, total_logit_loss / self.num_sub_heads, transformed_features

    def compute_average_entropy(self, probabilities, eps=1e-6):
        '''
        Computes the average entropy for the predicted probabilities
        :param probabilities: (batch_size, num_probabilities) tensor representing assignment probabilities
        :return: Average entropy across samples
        '''
        samples_count = probabilities.size()[0]

        # Computes entropy ensuring 0 probabilities are never passed to the logarithm
        entropy = -(probabilities * torch.log(torch.clamp(probabilities, eps, 1))).sum() / samples_count

        return entropy

    def filter_rows_by_threshold(self, tensor, threshold):
        tensor = tensor[(torch.max(tensor, dim=1)[0] >= threshold).type(torch.ByteTensor)]
        return tensor

    def compute_refinement_entropy_losses(self, training_batch, head, threshold):
        '''
        Performs computation of the entropy loss terms associated to refining for the test domain.
        :param training_batch: RefinementTrainBatch data for the current step
        :return:
        '''
        # Get current batches in the form (images_tensors, domain_id, ground_truth)
        domain_batch_plain, domain_batch_transformed = training_batch.get_batches()

        all_plain_cluster_probabilities, all_plain_logits, plain_features, _ = self.encoder_network(domain_batch_plain[0], domain_batch_plain[1], is_transformed=False, head=head)
        all_transformed_cluster_probabilities, all_transformed_logits, transformed_features, _ = self.encoder_network(domain_batch_transformed[0], domain_batch_transformed[1], is_transformed=True, head=head)

        total_entropy_loss = torch.tensor([0.0]).cuda()
        total_logit_loss = torch.tensor([0.0]).cuda()
        # Computes losses for each sub head
        for current_subhead_index in range(self.num_sub_heads):
            plain_cluster_probabilities = all_plain_cluster_probabilities[current_subhead_index]
            plain_logits = all_plain_logits[current_subhead_index]
            transformed_cluster_probabilities = all_transformed_cluster_probabilities[current_subhead_index]
            transformed_logits = all_transformed_logits[current_subhead_index]

            plain_cluster_probabilities = self.filter_rows_by_threshold(plain_cluster_probabilities, threshold)
            transformed_cluster_probabilities = self.filter_rows_by_threshold(transformed_cluster_probabilities, threshold)
            all_probabilities = torch.cat((plain_cluster_probabilities, transformed_cluster_probabilities))

            entropy = self.compute_average_entropy(all_probabilities)
            total_entropy_loss += entropy

            all_logits = torch.cat([plain_logits, transformed_logits])
            logit_loss = self.compute_logit_variance_loss(all_logits)

            total_logit_loss += logit_loss

        # Computes average losses over all subheads
        return total_entropy_loss / self.num_sub_heads, total_logit_loss / self.num_sub_heads, transformed_features

    def compute_cluster_losses(self, training_batch, head, current_epoch, per_domain_losses=True):
        '''
        Performs computation of the loss terms associated to clustering for the specified domain.
        :param training_batch: MixedDomainTrainBatch data for the current step
        :param per_domain_losses: if true, returns also an estimation of the losses for each domain
        :return:
        '''

        current_lamb_value = min(self.config.lamb, 1.0 + (self.config.lamb - 1.0) * current_epoch / self.config.lamb_annealing_epochs)

        # Get current batches in the form (images_tensors, domain_id, ground_truth)
        domain_batch_plain, domain_batch_transformed = training_batch.get_batches()

        all_plain_cluster_probabilities, all_plain_logits, plain_features, _ = self.encoder_network(domain_batch_plain[0], domain_batch_plain[1], is_transformed=False, head=head)
        all_transformed_cluster_probabilities, all_transformed_logits, transformed_features, _ = self.encoder_network(domain_batch_transformed[0], domain_batch_transformed[1], is_transformed=True, head=head)

        # Accumulator of losses divided by domain
        total_per_domain_mi_losses = torch.tensor([0.0] * self.domains_count).cuda()

        total_mi_loss = torch.tensor([0.0]).cuda()
        total_domain_mi_loss = torch.tensor([0.0]).cuda()
        total_logit_loss = torch.tensor([0.0]).cuda()
        total_centroid_loss = torch.tensor([0.0]).cuda()
        # Computes losses for each sub head
        for current_subhead_index in range(self.num_sub_heads):
            plain_cluster_probabilities = all_plain_cluster_probabilities[current_subhead_index]
            plain_logits = all_plain_logits[current_subhead_index]
            transformed_cluster_probabilities = all_transformed_cluster_probabilities[current_subhead_index]
            transformed_logits = all_transformed_logits[current_subhead_index]

            global_centroid_estimations, domain_centroids_estimations = self.centroid_estimators_map[head][current_subhead_index](torch.cat([plain_features, transformed_features]), torch.cat([domain_batch_plain[1], domain_batch_transformed[1]]), torch.cat([plain_cluster_probabilities, transformed_cluster_probabilities]))
            total_centroid_loss += self.compute_centroid_loss(global_centroid_estimations, domain_centroids_estimations)

            # Computes the loss for each domain
            if per_domain_losses:

                per_domain_plain_probabilities = self.create_domain_ranges(plain_cluster_probabilities, domain_batch_plain[1])
                per_domain_transformed_probabilities = self.create_domain_ranges(transformed_cluster_probabilities, domain_batch_transformed[1])

                assert(len(per_domain_plain_probabilities) == self.domains_count)
                assert(len(per_domain_transformed_probabilities) == self.domains_count)
                # Each domain is in the form (tensor, domain_id)
                for domain_idx, (current_domain_plain, current_domain_transformed) in enumerate(zip(per_domain_plain_probabilities, per_domain_transformed_probabilities)):
                    # Check probabilities correspond to the same domain
                    assert (current_domain_plain[1] == current_domain_transformed[1])
                    # Checks ids are ordered
                    assert (domain_idx == current_domain_plain[1])
                    per_domain_clustering_joint_probability = self.compute_cluster_joint(current_domain_plain[0], current_domain_transformed[0])

                    estimated_per_domain_clustering_joint_probability = self.domain_specific_estimators_map[(head, domain_idx)][current_subhead_index](per_domain_clustering_joint_probability)

                    current_domain_loss = -self.mutual_information(estimated_per_domain_clustering_joint_probability, current_lamb_value)
                    total_per_domain_mi_losses[current_domain_plain[1]] += current_domain_loss

            current_main_estimator = self.main_estimators_map[head][current_subhead_index]
            current_domain_estimator = self.domain_estimators_map[head][current_subhead_index]

            # Estimates transformed images probability
            clustering_joint_probability = self.compute_cluster_joint(plain_cluster_probabilities, transformed_cluster_probabilities)
            estimated_clustering_joint_probability = current_main_estimator(clustering_joint_probability)

            # Estimates domain-cluster assigment probabilities
            if not self.config.transformed_is_domain:
                # Compute for plain and transformed images then average
                domain_cluster_joint_probability_transformed = self.compute_domain_cluster_joint(domain_batch_transformed[1], transformed_cluster_probabilities)
                domain_cluster_joint_probability_plain = self.compute_domain_cluster_joint(domain_batch_plain[1], plain_cluster_probabilities)

                # Computes the domain loss using the probability matrix averaged between plain and transformed images
                if not self.config.domain_cluster_separate_transform:
                    domain_cluster_joint_probability = (domain_cluster_joint_probability_transformed + domain_cluster_joint_probability_plain) / 2
                    estimated_domain_cluster_joint_probability = current_domain_estimator(domain_cluster_joint_probability)
                    domain_mi_loss = self.mutual_information(estimated_domain_cluster_joint_probability, 1.0)
                # Computes the domain loss separately for the plain and transformed images
                else:
                    estimated_domain_cluster_joint_probability = current_domain_estimator(domain_cluster_joint_probability_plain)
                    estimated_domain_cluster_transformed_joint_probability = self.domain_estimators_transformed_map[head][current_subhead_index](domain_cluster_joint_probability_transformed)
                    domain_mi_loss = (self.mutual_information(estimated_domain_cluster_joint_probability, 1.0) + self.mutual_information(estimated_domain_cluster_transformed_joint_probability, 1.0)) / 2
            else:
                # Computes the domain joint probability using separate domains for the transformed images
                domain_cluster_joint_probability = self.compute_domain_cluster_joint(torch.cat((domain_batch_plain[1], domain_batch_transformed[1] + self.domains_count)), torch.cat((plain_cluster_probabilities, transformed_cluster_probabilities)))
                estimated_domain_cluster_joint_probability = current_domain_estimator(domain_cluster_joint_probability)
                domain_mi_loss = self.mutual_information(estimated_domain_cluster_joint_probability, 1.0)


            all_logits = torch.cat([plain_logits, transformed_logits])
            logit_loss = self.compute_logit_variance_loss(all_logits)


            # Maximize mutual information between transformed images, minimize mutual information between cluster assignments and domain information
            total_mi_loss += -self.mutual_information(estimated_clustering_joint_probability, current_lamb_value)
            total_domain_mi_loss += domain_mi_loss
            total_logit_loss += logit_loss

        # Computes average losses over all subheads
        return total_mi_loss / self.num_sub_heads, total_domain_mi_loss / self.num_sub_heads, total_logit_loss / self.num_sub_heads, transformed_features, total_per_domain_mi_losses / self.num_sub_heads, total_centroid_loss / self.num_sub_heads

    def compute_supervised_losses(self, training_batch):
        '''
        Computes the Cross entropy loss for the given training batch and input domain
        :param training_batch:
        :return:
        '''

        # Get current batches in the form (images_tensors, domain_id)
        _, domain_batch_transformed = training_batch.get_batches()
        images_tensor = domain_batch_transformed[0]
        domain_tensor = domain_batch_transformed[1]
        ground_truth_tensor = domain_batch_transformed[2]

        loss = torch.nn.CrossEntropyLoss()
        _, transformed_logits, _, _ = self.encoder_network(images_tensor, domain_tensor, is_transformed=False, head='B') # No need for IIC, so we don't have transformed images

        # Returns the average loss between subheads
        total_loss = torch.tensor([0.0]).cuda()
        for i in range(self.num_sub_heads):
            total_loss += loss(transformed_logits[i], ground_truth_tensor.type(torch.LongTensor).cuda())
        return total_loss / self.num_sub_heads

    def load(self, state_dict):
        '''
        Loads state from dictionary. The dictionary will contain all keys returned by calls to get_state

        :param state_dict:
        :return:
        '''
        self.encoder_network.load_state_dict(state_dict["encoder"])

        # Loads the estimators state

        for subhead_index in range(self.num_sub_heads):
            state_dict_main_a = state_dict["main_estimator_A_{}".format(subhead_index)]
            state_dict_main_b = state_dict["main_estimator_B_{}".format(subhead_index)]

            self.main_estimators_map['A'][subhead_index].load_state_dict(state_dict_main_a)
            self.main_estimators_map['B'][subhead_index].load_state_dict(state_dict_main_b)

            try:
                self.domain_estimators_map['A'][subhead_index].load_state_dict(state_dict["domain_estimator_A_{}".format(subhead_index)])
                self.domain_estimators_map['B'][subhead_index].load_state_dict(state_dict["domain_estimator_B_{}".format(subhead_index)])

                if self.config.domain_cluster_separate_transform:
                    self.domain_estimators_transformed_map['A'][subhead_index].load_state_dict(state_dict["domain_estimator_transformed_A_{}".format(subhead_index)])
                    self.domain_estimators_transformed_map['B'][subhead_index].load_state_dict(state_dict["domain_estimator_transformed_B_{}".format(subhead_index)])
            except:
                print("WARNING: Loading domain estimators failed. Did you load a model trained with a different number of domains? Reinitializaing module")

        for head in ('A', 'B'):
            for domain_id in range(self.domains_count):
                domain_specific_estimator = self.domain_specific_estimators_map[(head, domain_id)]
                for subhead_index in range(self.num_sub_heads):
                    key_name = "domain_specific_estimator_{}_{}_{}".format(head, domain_id, subhead_index)
                    # For code compatibility load state only if present in the state dictionary
                    if key_name in state_dict:
                        current_state = state_dict[key_name]
                        domain_specific_estimator[subhead_index].load_state_dict(current_state)

        for head in ('A', 'B'):
            for subhead_index in range(self.num_sub_heads):
                key_name = "centroid_estimator_{}_{}".format(head, subhead_index)
                # For code compatibility load state only if present in the state dictionary
                if key_name in state_dict:
                    self.centroid_estimators_map[head][subhead_index].load_state_dict(state_dict[key_name])


    def get_state(self):
        '''
        Gets the state dictionary representing the current state of the network
        :return:
        '''
        state_dict = {}

        state_dict["encoder"] = self.encoder_network.state_dict()

        for subhead_index in range(self.num_sub_heads):
            state_dict["main_estimator_A_{}".format(subhead_index)] = self.main_estimators_map['A'][subhead_index].state_dict()
            state_dict["main_estimator_B_{}".format(subhead_index)] = self.main_estimators_map['B'][subhead_index].state_dict()
            state_dict["domain_estimator_A_{}".format(subhead_index)] = self.domain_estimators_map['A'][subhead_index].state_dict()
            state_dict["domain_estimator_B_{}".format(subhead_index)] = self.domain_estimators_map['B'][subhead_index].state_dict()

            state_dict["domain_estimator_transformed_A_{}".format(subhead_index)] = self.domain_estimators_transformed_map['A'][subhead_index].state_dict()
            state_dict["domain_estimator_transformed_B_{}".format(subhead_index)] = self.domain_estimators_transformed_map['B'][subhead_index].state_dict()

        # Saves domain specific estimators
        for head in ('A', 'B'):
            for domain_id in range(self.domains_count):
                domain_specific_estimator = self.domain_specific_estimators_map[(head, domain_id)]
                for subhead_index in range(self.num_sub_heads):
                    current_estimator = domain_specific_estimator[subhead_index]
                    state_dict["domain_specific_estimator_{}_{}_{}".format(head, domain_id, subhead_index)] = current_estimator.state_dict()

        for head in ('A', 'B'):
            for subhead_index in range(self.num_sub_heads):
                state_dict["centroid_estimator_{}_{}".format(head, subhead_index)] = self.centroid_estimators_map[head][subhead_index].state_dict()

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

        for subhead_index in range(self.num_sub_heads):
            for current_estimators in self.main_estimators_map.values():
                current_estimators[subhead_index].cuda()
            for current_estimators in self.domain_estimators_map.values():
                current_estimators[subhead_index].cuda()

            for current_estimators in self.domain_estimators_transformed_map.values():
                current_estimators[subhead_index].cuda()

        for head in ('A', 'B'):
            for domain_id in range(self.domains_count):
                domain_specific_estimator = self.domain_specific_estimators_map[(head, domain_id)]
                for subhead_index in range(self.num_sub_heads):
                    domain_specific_estimator[subhead_index].cuda()

        for head in ('A', 'B'):
            for subhead_index in range(self.num_sub_heads):
                self.centroid_estimators_map[head][subhead_index].cuda()
