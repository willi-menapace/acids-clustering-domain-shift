import torch

from codebase.utils.cluster.management.basic_dial_evaluation_manager import BasicDialEvaluationManager


def list_collate_batcher(batch):
    return [element for element in batch]

class RefinementEvaluationManager(BasicDialEvaluationManager):
    '''
    Utility class for model evaluation
    '''

    def __init__(self, config, network_manager, processing_pool):
        super(RefinementEvaluationManager, self).__init__(config, network_manager, processing_pool)

    def perform_clustering(self, domain_id):
        '''
        Performs clustering of the given domain
        :param domain_id: id of the domain to cluster
        :return: list of (domain_length, 1) tensor with cluster assignment for each subhead,
                 (domain_length, 1) tensor with ground truth information
        '''
        dataloader = self.domain_dataloaders[domain_id]

        all_ground_truth = []
        all_features = []
        all_cluster_assignments = [[] for _ in range(self.num_sub_heads)] # An empty list for each subhead

        for dataloader_batch in dataloader:
            images_tensor, domain_id_tensor, ground_truth_tensor = self.build_evaluation_batch(dataloader_batch, domain_id)

            # Performs clustering
            _, _, features, _ = self.network_manager.encoder_network(images_tensor, domain_id_tensor)
            clusters_probabilities, _ = self.network_manager.heads(features, domain_id, 'B')

            all_ground_truth.append(ground_truth_tensor)
            all_features.append(features)

            # Adds clustering results for each subhead
            for subhead_index in range(self.num_sub_heads):
                all_cluster_assignments[subhead_index].append(torch.argmax(clusters_probabilities[subhead_index], dim=1).to(dtype=torch.int32))  # A result is produces for each subhead, return only the first

        return [torch.cat(subhead_cluster_assignments) for subhead_cluster_assignments in all_cluster_assignments], torch.cat(all_ground_truth), torch.cat(all_features)