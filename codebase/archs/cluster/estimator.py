import torch
import torch.nn as nn

class FixedMatrixEstimator(nn.Module):

    def __init__(self, rows, columns, initial_alpha=0.9, initial_value=None):
        '''
        Initializes the joint probability estimator for a (rows, columns) matrix with the given fixed alpha factor
        :param rows, columns: Dimension of the probability matrix to estimate
        :param initial_alpha: Value to use assign for alpha
        '''
        super(FixedMatrixEstimator, self).__init__()

        self.alpha = initial_alpha

        # Initializes the joint matrix as a uniform, independent distribution. Does not allow backpropagation to this parameter
        if initial_value is None:
            initial_value = torch.tensor([[1.0 / (rows * columns)] * columns] * rows, dtype=torch.float32)
        self.estimated_matrix = nn.Parameter(initial_value, requires_grad=False)

    def forward(self, latest_probability_matrix):
        return_matrix = self.estimated_matrix * self.alpha + latest_probability_matrix * (1 - self.alpha)

        # The estimated matrix must be detached from the backpropagation graph to avoid exhaustion of GPU memory
        self.estimated_matrix.data = return_matrix.detach()

        return return_matrix

class LearnableMatrixEstimator(nn.Module):

    def __init__(self, k, initial_alpha=0.9):
        '''
        Initializes the joint probability estimator for a (k, k) matrix with the given initial alpha factor
        :param k: Dimension of the square probability matrix to estimate
        :param initial_alpha: Initial value to assign to alpha
        '''
        super(LearnableMatrixEstimator, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(initial_alpha, dtype=torch.float32, requires_grad=True))

        # Initializes the joint matrix as a uniform, independent distribution. Does not allow backpropagation to this parameter
        self.estimated_matrix = nn.Parameter(torch.tensor([[1.0 / (k**2)] * k] * k, dtype=torch.float32), requires_grad=False)

    def forward(self, latest_probability_matrix):
        self.return_matrix = (torch.sigmoid(self.alpha)) * self.estimated_matrix + (-torch.sigmoid(self.alpha) + 1.0) * latest_probability_matrix

        # The estimated matrix must be detached from the backpropagation graph to avoid exhaustion of GPU memory
        self.estimated_matrix.data = self.return_matrix.detach()


        return self.return_matrix

class CentroidEstimator(nn.Module):
    '''
    Estimator for cluster centroids positions across domains
    '''

    def __init__(self, config, num_clusters):
        '''
        Initializes the centroid estimator based on the provided configuration and number of clusters that will need to
        be estimated
        :param config: configuration file
        :param num_clusters: number of clusters for which to estimate centroids
        '''
        super(CentroidEstimator, self).__init__()

        self.domains_count = config.domains_count
        self.features_size = config.encoder_features_count
        self.num_clusters = num_clusters
        self.alpha = config.centroid_estimation_alpha

        # Global estimator computed on points from every domain
        self.global_centroids_estimator = FixedMatrixEstimator(self.features_size, self.num_clusters, self.alpha, initial_value=torch.zeros((self.features_size, self.num_clusters)))
        # Estimator for centroids inside each domain
        self.domain_centroids_estimators = nn.ModuleList([FixedMatrixEstimator(self.features_size, self.num_clusters, self.alpha, initial_value=torch.zeros((self.features_size, self.num_clusters))) for i in range(self.domains_count)])

    def compute_centroids(self, numerator : torch.Tensor, cluster_probabilities : torch.Tensor, eps=0.001):
        '''
        Computes the centroids over the whole numerator and cluster probabilities tensors
        :param numerator: (num_points, features_size, num_clusters) tensor
        :param cluster_probabilities: (num_points, num_clusters) tensor
        :return: (features_size, num_clusters) tensor representing the centroids for each cluster
        '''

        numerator = torch.sum(numerator, dim=0, keepdim=False)
        cluster_probabilities = torch.sum(cluster_probabilities, dim=0, keepdim=False)
        cluster_probabilities = cluster_probabilities + eps
        return numerator / cluster_probabilities

    def forward(self, features : torch.Tensor, domains : torch.Tensor, cluster_probabilities : torch.Tensor):
        '''

        :param features: (batch_size, features_size) tensor of features to which to estimate centroids
        :param domains: (1, batch_size) tensor representing the domain of each sample
        :param cluster_probabilities: (batch_size, num_clusters) tensor of cluster assigment probabilities
        :return: global_centroid, domain_centroids_list where global centroid is a (feaures_size, num_clusters)
                 tensor representing the global centroids and domain_centroids_list is a list of length domains_count
                 with such centroid tensors computed on each domain
        '''

        unsqueezed_features = features.unsqueeze(-1)
        unsqueezed_cluster_probabilities = cluster_probabilities.unsqueeze(1)

        unsummed_numerator = unsqueezed_features * unsqueezed_cluster_probabilities

        global_centroids = self.compute_centroids(unsummed_numerator, unsqueezed_cluster_probabilities)
        estimated_global_centroids = self.global_centroids_estimator(global_centroids)

        estimated_domain_centroids = []
        for domain_idx in range(self.domains_count):
            # Extracts indices of elements of the current domain
            slice_indexes = torch.nonzero((domains.view(-1) == domain_idx))
            domain_numerator = unsummed_numerator[slice_indexes].squeeze(1)
            domain_cluster_probabilities = unsqueezed_cluster_probabilities[slice_indexes].squeeze(1)

            domain_centroid = self.compute_centroids(domain_numerator, domain_cluster_probabilities)
            current_domain_centroids = self.domain_centroids_estimators[domain_idx](domain_centroid)
            estimated_domain_centroids.append(current_domain_centroids)

        return estimated_global_centroids, estimated_domain_centroids
