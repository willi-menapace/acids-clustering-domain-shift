import torch
import torchvision
import numpy as np
import scipy
import json
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt

from codebase.utils.cluster.cluster_eval import compute_cluster_confusion_matrix
from codebase.utils.cluster.dict_wrapper import DictWrapper
from codebase.utils.cluster.eval_metrics import _hungarian_match
from codebase.utils.cluster.transforms import sobel_make_transforms, sobel_process, sobel_make_multi_transforms


def list_collate_batcher(batch):
    return [element for element in batch]

class BasicDialEvaluationManager:
    '''
    Utility class for model evaluation
    '''

    def __init__(self, config, network_manager, processing_pool):

        self.config = config
        self.domains_count = config.domains_count
        self.test_domains_count = config.test_domains_count
        self.network_manager = network_manager
        self.processing_pool = processing_pool
        self.num_sub_heads = config.num_sub_heads

        if hasattr(config, "transforms"):
            _, _, tf3 = sobel_make_multi_transforms(self.config)
        else:
            _, _, tf3 = sobel_make_transforms(self.config)

        self.transform = tf3

        self.all_domains_count = config.all_domains_count
        self.domain_dataloaders = [self.create_dataloader_from_path(domain_path) for domain_path in config.domain_datasets] + \
                                  [self.create_dataloader_from_path(domain_path) for domain_path in config.test_datasets]

        self.full_domain_dataloaders = []
        if len(config.test_datasets_full) > 0:
            self.full_domain_dataloaders = [self.create_dataloader_from_path(domain_path) for domain_path in config.test_datasets_full]

        self.latest_domain_features = [None] * self.all_domains_count
        self.latest_predicted_clusters = [None] * self.all_domains_count
        self.ground_truth = [None] * self.all_domains_count

    def create_dataloader_from_path(self, path, workers=0):
        dataset = torchvision.datasets.ImageFolder(
            root=path)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.config.batch_sz,
                                                 # full batch
                                                 shuffle=False,
                                                 collate_fn=list_collate_batcher,
                                                 num_workers=workers,
                                                 drop_last=False)

        return dataloader

    def build_evaluation_batch(self, dataloader_batch, domain_id):
        '''
        Builds evaluation tensors starting from a [(PIL image, gt class), ...] batch
        :param dataloader_batch:
        :param domain_id:
        :return: images_tensor, domain_id_tensor, tensor with ground truth labels
        '''

        # Domain id that will be passed to the network. May be different from the original domain id in case
        # of remapping
        substituted_domain = domain_id
        is_test_domain = domain_id >= self.config.domains_count
        if is_test_domain and len(self.config.map_test_to_domains) > 0:
            # Determines the id to substitute at test time according to the user provided map
            test_domain_id = domain_id - self.config.domains_count
            substituted_domain = self.config.map_test_to_domains[test_domain_id]

        domain_id_tensor = torch.zeros((len(dataloader_batch), 1), dtype=torch.int32).fill_(substituted_domain).cuda()
        ground_truth_tensor = torch.IntTensor([element[1] for element in dataloader_batch]).cuda()

        images = [row[0] for row in dataloader_batch]

        transformed_list = self.processing_pool.map(self.transform, images)

        images_tensor = torch.stack(transformed_list).cuda()
        if not self.config.use_rgb:
            images_tensor = sobel_process(images_tensor, include_rgb=False)

        return images_tensor, domain_id_tensor, ground_truth_tensor

    def perform_clustering(self, domain_id):
        '''
        Performs clustering of the given domain
        :param domain_id: id of the domain to cluster
        :return: list of (domain_length, 1) tensor with cluster assignment for each subhead,
                 (domain_length, 1) tensor with ground truth information
        '''
        dataloader = self.domain_dataloaders[domain_id]
        # if domain is a test domain, there are full domains for test and we are in evaluation then use the full domains
        if domain_id >= self.domains_count and self.full_domain_dataloaders and not self.network_manager.encoder_network.training:
            dataloader = self.full_domain_dataloaders[domain_id - self.domains_count]

        all_ground_truth = []
        all_features = []
        all_cluster_assignments = [[] for _ in range(self.num_sub_heads)] # An empty list for each subhead

        for dataloader_batch in dataloader:
            images_tensor, domain_id_tensor, ground_truth_tensor = self.build_evaluation_batch(dataloader_batch, domain_id)

            # Performs clustering
            clusters_probabilities, _, features, _ = self.network_manager.encoder_network(images_tensor, domain_id_tensor, is_transformed=False)

            all_ground_truth.append(ground_truth_tensor)
            all_features.append(features)

            # Adds clustering results for each subhead
            for subhead_index in range(self.num_sub_heads):
                all_cluster_assignments[subhead_index].append(torch.argmax(clusters_probabilities[subhead_index], dim=1).to(dtype=torch.int32))  # A result is produces for each subhead, return only the first

        return [torch.cat(subhead_cluster_assignments) for subhead_cluster_assignments in all_cluster_assignments], torch.cat(all_ground_truth), torch.cat(all_features)

    def compute_accuracy(self, predicted_clusters, ground_truth_clusters):
        '''
        Computes accuracy of the cluster assignment
        Can be called only if the number of clusters is the same as the ground truth clusters
        :param predicted_clusters:
        :param ground_truth_clusters:
        :return: Cluster accuracy score
        '''
        assert(self.config.gt_k == self.config.output_k_B)
        num_samples = predicted_clusters.shape[0]

        reordered_preds = self.class_predictions_to_ground_truth(predicted_clusters, ground_truth_clusters)

        accuracy = int((reordered_preds == ground_truth_clusters).sum()) / float(num_samples)
        return accuracy

    def class_predictions_to_ground_truth(self, predicted_clusters, ground_truth_clusters):
        '''
        Computes predicted ground truth class of the cluster assignment
        Can be called only if the number of clusters is the same as the ground truth clusters
        :param predicted_clusters:
        :param ground_truth_clusters:
        :return: tensor of predictions translated to ground truth labels
        '''

        num_samples = predicted_clusters.shape[0]

        match = _hungarian_match(predicted_clusters, ground_truth_clusters, self.config.gt_k, self.config.output_k_B)

        found = torch.zeros(self.config.gt_k)
        reordered_preds = torch.zeros(num_samples, dtype=predicted_clusters.dtype).cuda()

        for pred_i, target_i in match:
            # reordered_preds[flat_predss_all[i] == pred_i] = target_i
            reordered_preds[torch.eq(predicted_clusters, int(pred_i))] = torch.from_numpy(
                np.array(target_i)).cuda().int().item()
            found[pred_i] = 1
        assert (found.sum() == self.config.gt_k)  # each output_kz must get mapped

        return reordered_preds

    def compute_latest_features_visualization(self, savepath):


        domain_ids = []
        for domain_id in range(self.all_domains_count):
            current_features = self.latest_domain_features[domain_id]
            rows, cols = np.shape(current_features)
            domain_ids.extend([domain_id] * rows)
        domain_ids = np.asarray(domain_ids)

        allfeatures = np.concatenate(self.latest_domain_features, axis=0)

        # Performs dimensionality reduction
        embeddings = TSNE(n_jobs=14).fit_transform(allfeatures)

        # Plots domains distribution
        vis_x = embeddings[:, 0]
        vis_y = embeddings[:, 1]
        if self.all_domains_count > 1:
            plt.scatter(vis_x, vis_y, c=domain_ids, cmap=plt.cm.get_cmap("jet", self.all_domains_count), marker='.', alpha=0.2)
            plt.colorbar(ticks=range(self.all_domains_count))
            plt.savefig(savepath + "_domains.pdf")
            plt.close()

        all_ground_truth = np.concatenate(self.ground_truth, axis=0)

        # Plots ground truth clusters
        plt.scatter(vis_x, vis_y, c=all_ground_truth, cmap=plt.cm.get_cmap("jet", self.config.gt_k), marker='.', alpha=0.2)
        plt.colorbar(ticks=range(self.config.gt_k))
        plt.savefig(savepath + "_gt.pdf")
        plt.close()

        for domain_idx in range(self.all_domains_count):
            # Extracts features and ground truth for the current domain
            current_embeddings = embeddings[domain_ids == domain_idx]
            ground_truth = all_ground_truth[domain_ids == domain_idx]

            plt.scatter(current_embeddings[:, 0], current_embeddings[:, 1], c=ground_truth, cmap=plt.cm.get_cmap("jet", self.config.gt_k), marker='.', alpha=0.2)
            plt.colorbar(ticks=range(self.config.gt_k))
            plt.savefig(savepath + "_gt_domain_{}.pdf".format(domain_idx))
            plt.close()

        for subhead_index in range(self.num_sub_heads):
            # Concatenates all domain features for the current subhead
            all_predicted_clusters = np.concatenate([self.latest_predicted_clusters[domain][subhead_index] for domain in range(self.all_domains_count)], axis=0)

            # Plots ground truth clusters
            plt.scatter(vis_x, vis_y, c=all_predicted_clusters, cmap=plt.cm.get_cmap("jet", self.config.gt_k), marker='.', alpha=0.2)
            plt.colorbar(ticks=range(self.config.gt_k))
            plt.savefig(savepath + "_clusters_subhead_{}.pdf".format(subhead_index))
            plt.close()

    def get_domain_distribution_by_cluster(self, cluster_id, subhead_id):
        '''
        Returns the probability distribution of domains inside a specified cluster
        :param cluster_id:
        :param subhead_id: id of the subhead to condider for clustering
        :return:
        '''
        domain_distribution = np.zeros(self.all_domains_count, dtype=np.float)
        for domain_id in range(self.all_domains_count):
            clusters = self.latest_predicted_clusters[domain_id][subhead_id]
            domain_length = np.shape(clusters[clusters == cluster_id])[0]
            domain_distribution[domain_id] = domain_length
        domain_distribution = domain_distribution / domain_distribution.sum()
        return domain_distribution

    def get_domain_distribution(self):
        '''
        Returns the probability distribution of all domains
        :return:
        '''
        domain_distribution = np.zeros(self.all_domains_count, dtype=np.float)
        for domain_id in range(self.all_domains_count):
            clusters = self.latest_predicted_clusters[domain_id][0]
            domain_length = np.shape(clusters)[0]
            domain_distribution[domain_id] = domain_length
        domain_distribution = domain_distribution / domain_distribution.sum()
        return domain_distribution

    def get_clusters_distribution(self, subhead_id):
        '''
        Returns the probability distribution of clusters produced by a certain head across all domains
        :subhead_id:
        :return:
        '''
        all_predicted_clusters = np.concatenate([self.latest_predicted_clusters[domain_id][subhead_id] for domain_id in range(self.all_domains_count)], axis=0)
        clusters_distribution = np.zeros(self.config.output_k_B, dtype=np.float)

        for cluster_id in range(self.config.output_k_B):
            cluster_length = np.shape(all_predicted_clusters[all_predicted_clusters == cluster_id])[0]
            clusters_distribution[cluster_id] = cluster_length
        clusters_distribution = clusters_distribution / clusters_distribution.sum()
        return clusters_distribution

    def mutual_information_domains_clusters(self):
        '''
        Computes mutual information between domains and domains|cluster_assignment
        :return: map containing mutual information as well as domain entropy and conditional domain entropy given clusters
        '''

        result = {
            "domain_entropy": [],
            "domain_entropy_given_cluster": [],
            "mutual_information": []
        }

        domain_distribution = self.get_domain_distribution()

        for subhead_id in range(self.num_sub_heads):

            domain_distribution_by_cluster = [self.get_domain_distribution_by_cluster(cluster_id, subhead_id) for cluster_id in range(self.config.output_k_B)]
            cluster_distribution = self.get_clusters_distribution(subhead_id)

            entropy_domain_given_cluster = 0.0
            for cluster_id in range(self.config.output_k_B):
                # Avoids considering empty clusters
                if cluster_distribution[cluster_id] > 0.0001:
                    entropy_domain_given_cluster += cluster_distribution[cluster_id] * scipy.stats.entropy(domain_distribution_by_cluster[cluster_id])
            domain_entropy = scipy.stats.entropy(domain_distribution)
            mutual_information = domain_entropy - entropy_domain_given_cluster

            result["domain_entropy"].append(domain_entropy)
            result["domain_entropy_given_cluster"].append(entropy_domain_given_cluster)
            result["mutual_information"].append(mutual_information)

        return result

    def evaluate(self, domain_id):
        '''
        Evaluates clustering performance of the given domain

        :param domain_id: id of the domain of which to compute performance
        :return:
        '''

        results_map = {
            "nmi": [],
            "mutual_information": [],
            "conditional_entropy": [],
            "ground_truth_clusters_entropy": [],
            "predicted_clusters_entropy": [],
            "accuracy": [],
            "confusion_matrix": []
        }

        predicted_clusters, ground_truth_clusters, features = self.perform_clustering(domain_id)

        self.latest_domain_features[domain_id] = features.cpu().numpy()
        self.latest_predicted_clusters[domain_id] = [] # Predicted clusters are different for each domain

        self.ground_truth[domain_id] = ground_truth_clusters.cpu().numpy()

        ground_truth_clusters_npy = ground_truth_clusters.cpu().numpy()

        for subhead_index in range(self.num_sub_heads):

            # Computes accuracy if the number of predicted clusters matches the number of ground truth ones
            accuracy = None
            if self.config.gt_k == self.config.output_k_B:
                accuracy = self.compute_accuracy(predicted_clusters[subhead_index], ground_truth_clusters)

            subhead_predicted_clusters = predicted_clusters[subhead_index].cpu().numpy()
            self.latest_predicted_clusters[domain_id].append(subhead_predicted_clusters)

            confusion_matrix = compute_cluster_confusion_matrix(subhead_predicted_clusters, ground_truth_clusters_npy,
                                                                self.config.output_k_B,
                                                                self.config.gt_k)

            # Computes entropies
            _, predicted_clusters_distribution = np.unique(subhead_predicted_clusters, return_counts=True)
            predicted_clusters_entropy = scipy.stats.entropy(predicted_clusters_distribution)

            _, ground_truth_clusters_distribution = np.unique(ground_truth_clusters_npy, return_counts=True)
            ground_truth_clusters_entropy = scipy.stats.entropy(ground_truth_clusters_distribution)

            # Computes information scores
            mutual_information = mutual_info_score(subhead_predicted_clusters, ground_truth_clusters_npy)
            conditional_entropy = -(mutual_information - ground_truth_clusters_entropy)
            nmi = normalized_mutual_info_score(subhead_predicted_clusters, ground_truth_clusters_npy)

            results_map["nmi"].append(nmi)
            results_map["mutual_information"].append(mutual_information)
            results_map["conditional_entropy"].append(conditional_entropy)
            results_map["ground_truth_clusters_entropy"].append(ground_truth_clusters_entropy)
            results_map["predicted_clusters_entropy"].append(predicted_clusters_entropy)
            results_map["accuracy"].append(accuracy)
            results_map["confusion_matrix"].append(confusion_matrix)

        return results_map

    def compute_clustering_information(self, domain_id, subhead_index=0):
        '''
        Computes foreach image the predicted ground truth label

        :param domain_id:
        :return:
        '''

        predicted_clusters, ground_truth, features = self.perform_clustering(domain_id)
        ground_truth_npy = ground_truth.cpu().numpy()

        predicted_ground_truth = self.class_predictions_to_ground_truth(predicted_clusters[subhead_index], ground_truth).cpu().numpy()

        dataloader = self.domain_dataloaders[domain_id]
        image_paths = [entry[0] for entry in dataloader.dataset.imgs]

        results = {}
        for index in range(len(image_paths)):
            ground_truth_label = ground_truth_npy[index]
            predicted_label = predicted_ground_truth[index]
            if predicted_label not in results:
                results[predicted_label] = []

            results[predicted_label].append((image_paths[index], predicted_label, ground_truth_label))

        return results

if __name__ == "__main__":
    config = None
    with open("config/basic_dial.json") as json_file:
        config = json.load(json_file)
        config = DictWrapper(config)

    eval_manager = BasicDialEvaluationManager(config, None, None)
    for i in range(eval_manager.all_domains_count):
        eval_manager.latest_domain_features[i] = np.random.rand(100, 10)
        eval_manager.ground_truth[i] = np.random.randint(0, 64, (100))

    eval_manager.compute_latest_features_visualization("/home/willi/visualization")