import torch

from codebase.utils.cluster.management.basic_dial_evaluation_manager import BasicDialEvaluationManager


class ContinuousDAEvaluationManager(BasicDialEvaluationManager):

    def __init__(self, config, network_manager, optimizer_manager, processing_pool):
        super(ContinuousDAEvaluationManager, self).__init__(config, network_manager, processing_pool)

        self.optimizer_manager = optimizer_manager

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

    def compute_continuous_da_entropy_losses(self, images_tensor, domain_id_tensor, ground_truth_tensor, threshold):
        '''
        Performs computation of the entropy loss terms associated to refining for the test domain.

        :return:
        '''

        all_plain_cluster_probabilities, all_plain_logits, plain_features, _ = self.network_manager.encoder_network(images_tensor, domain_id_tensor, is_transformed=False, head='B')

        total_entropy_loss = torch.tensor([0.0]).cuda()
        # Computes losses for each sub head
        for current_subhead_index in range(self.num_sub_heads):
            plain_cluster_probabilities = all_plain_cluster_probabilities[current_subhead_index]

            plain_cluster_probabilities = self.filter_rows_by_threshold(plain_cluster_probabilities, threshold)

            entropy = self.compute_average_entropy(plain_cluster_probabilities)

            total_entropy_loss += entropy

        # Computes average losses over all subheads
        return total_entropy_loss / self.num_sub_heads

    def perform_clustering(self, domain_id):
        '''
        Performs clustering of the given domain. Network must be in training mode
        :param domain_id: id of the domain to cluster
        :return: list of (domain_length, 1) tensor with cluster assignment for each subhead,
                 (domain_length, 1) tensor with ground truth information
        '''

        # The function will change the state of the network to perform continual learning, so we require the network to be in train mode
        if not self.network_manager.encoder_network.training:
            raise Exception("Continual learning modifies the network parameters, but network is not in training mode. Pass a network in traning mode to perform evaluation.")

        dataloader = self.domain_dataloaders[domain_id]
        # if domain is a test domain, there are full domains for test and we are in evaluation then use the full domains
        if domain_id >= self.domains_count and self.full_domain_dataloaders and not self.network_manager.encoder_network.training:
            assert(False) # We cannot use continuous domain adaptation just for evaluation on a full domain after training on a fewshot domain
            dataloader = self.full_domain_dataloaders[domain_id - self.domains_count]

        all_ground_truth = []
        all_features = []
        all_cluster_assignments = [[] for _ in range(self.num_sub_heads)] # An empty list for each subhead

        for dataloader_batch in dataloader:
            images_tensor, domain_id_tensor, ground_truth_tensor = self.build_evaluation_batch(dataloader_batch, domain_id)

            # Performs optimization on the current batch
            self.network_manager.train()
            loss = self.compute_continuous_da_entropy_losses(images_tensor, domain_id_tensor, ground_truth_tensor, self.config.refinement_entropy_threshold)
            self.optimizer_manager.encoder_optimizer.zero_grad()
            loss.backward()
            self.optimizer_manager.encoder_optimizer.step()
            self.network_manager.eval()

            print(" - Continuous DA loss: {}".format(loss.item()))
            with torch.no_grad():
                # Performs clustering
                clusters_probabilities, _, features, _ = self.network_manager.encoder_network(images_tensor, domain_id_tensor, is_transformed=False)

                all_ground_truth.append(ground_truth_tensor)
                all_features.append(features)

                # Adds clustering results for each subhead
                for subhead_index in range(self.num_sub_heads):
                    all_cluster_assignments[subhead_index].append(torch.argmax(clusters_probabilities[subhead_index], dim=1).to(dtype=torch.int32))  # A result is produces for each subhead, return only the first

        self.network_manager.train() # Sets the network again in training mode
        return [torch.cat(subhead_cluster_assignments) for subhead_cluster_assignments in all_cluster_assignments], torch.cat(all_ground_truth), torch.cat(all_features)
