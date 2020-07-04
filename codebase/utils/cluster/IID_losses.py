import sys

import torch
import torch.nn as nn
import numpy as np


def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    # has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                             k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j) \
                      - lamb * torch.log(p_j) \
                      - lamb * torch.log(p_i))

    loss = loss.sum()

    loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
                              - torch.log(p_j) \
                              - torch.log(p_i))

    loss_no_lamb = loss_no_lamb.sum()

    return loss, loss_no_lamb


def IID_plus_loss(plain_output, transformed_output, joint_probabilities_estimations, lamb=1.0,
                  EPS=sys.float_info.epsilon):
    # has had softmax applied
    _, k = plain_output.size()

    joint_probability_matrix = compute_joint_with_estimations(plain_output, transformed_output,
                                                              joint_probabilities_estimations)

    assert (joint_probability_matrix.size() == (k, k))

    marginal_i = joint_probability_matrix.sum(dim=1).view(k, 1).expand(k, k)
    marginal_j = joint_probability_matrix.sum(dim=0).view(1, k).expand(k,
                                                                       k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    joint_probability_matrix[(joint_probability_matrix < EPS).data] = EPS
    marginal_j[(marginal_j < EPS).data] = EPS
    marginal_i[(marginal_i < EPS).data] = EPS

    loss = - joint_probability_matrix * (torch.log(joint_probability_matrix) \
                                         - lamb * torch.log(marginal_j) \
                                         - lamb * torch.log(marginal_i))

    loss = loss.sum()

    loss_no_lamb = - joint_probability_matrix * (torch.log(joint_probability_matrix) \
                                                 - torch.log(marginal_j) \
                                                 - torch.log(marginal_i))

    loss_no_lamb = loss_no_lamb.sum()

    return loss, loss_no_lamb

def IID_alpha_plus_loss(plain_output, transformed_output, old_joint_probabilities, alpha, lamb=1.0, EPS=sys.float_info.epsilon):
    # has had softmax applied
    _, k = plain_output.size()

    joint_probability_matrix = compute_joint(plain_output, transformed_output)

    # If the batch is not the first use an exponential average to estimate the probability matrix
    if old_joint_probabilities is not None:
        joint_probability_matrix = (1.0 - alpha) * joint_probability_matrix + alpha * old_joint_probabilities

    assert (joint_probability_matrix.size() == (k, k))

    marginal_i = joint_probability_matrix.sum(dim=1).view(k, 1).expand(k, k)
    marginal_j = joint_probability_matrix.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    joint_probability_matrix[(joint_probability_matrix < EPS).data] = EPS
    marginal_j[(marginal_j < EPS).data] = EPS
    marginal_i[(marginal_i < EPS).data] = EPS

    loss = - joint_probability_matrix * (torch.log(joint_probability_matrix) \
                                         - lamb * torch.log(marginal_j) \
                                         - lamb * torch.log(marginal_i))

    loss = loss.sum()

    loss_no_lamb = - joint_probability_matrix * (torch.log(joint_probability_matrix) \
                                                 - torch.log(marginal_j) \
                                                 - torch.log(marginal_i))

    loss_no_lamb = loss_no_lamb.sum()

    return loss, loss_no_lamb, joint_probability_matrix.detach()

def IID_learnable_alpha_plus_loss(joint_probability_matrix, lamb=1.0, EPS=sys.float_info.epsilon):
    k = joint_probability_matrix.size()[0]

    assert (joint_probability_matrix.size() == (k, k))

    marginal_i = joint_probability_matrix.sum(dim=1).view(k, 1).expand(k, k)
    marginal_j = joint_probability_matrix.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    joint_probability_matrix[(joint_probability_matrix < EPS).data] = EPS
    marginal_j[(marginal_j < EPS).data] = EPS
    marginal_i[(marginal_i < EPS).data] = EPS

    loss = - joint_probability_matrix * (torch.log(joint_probability_matrix) \
                                         - lamb * torch.log(marginal_j) \
                                         - lamb * torch.log(marginal_i))

    loss = loss.sum()

    loss_no_lamb = - joint_probability_matrix * (torch.log(joint_probability_matrix) \
                                                 - torch.log(marginal_j) \
                                                 - torch.log(marginal_i))

    loss_no_lamb = loss_no_lamb.sum()

    return loss, loss_no_lamb

def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


def compute_joint_with_estimations(plain_output, transformed_output, joint_probabilities_estimations):
    '''
    Computes a smoothed joint probabilities matrix starting from the current outputs and a sequence of sums of estimated probability matrices

    :param plain_output: the plain network outputs
    :param transformed_output: the transformed network outputs
    :param joint_probabilities_estimations: list of sums of estimated probabilities matrices
    :return: estimated probability matrix
    '''

    # Computes sum of the current joints
    joint_probabilities_sum = plain_output.unsqueeze(2) * transformed_output.unsqueeze(1)  # bn, k, k
    joint_probabilities_sum = joint_probabilities_sum.sum(dim=0)  # k, k

    # Computes sum of the estimated joints
    if len(joint_probabilities_estimations) > 0:
        estimated_joint_probabilities_sum = torch.stack(joint_probabilities_estimations).sum(0)
    else:
        estimated_joint_probabilities_sum = torch.zeros_like(joint_probabilities_sum)
      
    # Merges the sums and normalizes
    joint_probabilities_sum += estimated_joint_probabilities_sum

    joint_probabilities_sum = (joint_probabilities_sum + joint_probabilities_sum.t()) / 2.  # symmetrise
    joint_probabilities = joint_probabilities_sum / joint_probabilities_sum.sum()  # normalise

    #print(joint_probabilities.trace())
    #print(joint_probabilities)

    return joint_probabilities


def intra_distance_loss(cluster_probabilities, features, delta=0.5):
    """
    Computes an intra cluster distance loss based on feature distance minimization
    """

    cluster_probabilities_sum = cluster_probabilities.sum(dim=0)

    cluster_features_prod = torch.unsqueeze(cluster_probabilities, dim=-1) * torch.unsqueeze(features, dim=1)
    cluster_features_sum = cluster_features_prod.sum(dim=0)
    cluster_features_sum = cluster_features_sum.transpose(0, 1)

    avg_features_per_cluster = cluster_features_sum / (cluster_probabilities_sum)
    # avg_features_per_cluster_nod = cluster_features_sum / (cluster_probabilities_sum)

    squared_features_per_cluster = (avg_features_per_cluster - torch.unsqueeze(features, dim=-1)) ** 2
    squared_distances_per_cluster_pre = squared_features_per_cluster.sum(dim=1)

    squared_distances_per_cluster = (squared_distances_per_cluster_pre * cluster_probabilities).sum(dim=0) / (
            delta + cluster_probabilities_sum)
    # squared_distances_per_cluster_nod = (squared_distances_per_cluster_pre * cluster_probabilities).sum(dim=0) / (cluster_probabilities_sum)

    return torch.mean(squared_distances_per_cluster)


def IID_trace_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    # has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k,
                                             k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    # print(p_i_j.cpu().deach().numpy())
    loss = - p_i_j * (torch.log(p_i_j) \
                      - lamb * torch.log(p_j) \
                      - lamb * torch.log(p_i))

    loss = loss.sum()

    loss = loss + 0.5 * (1 - p_i_j.trace())

    loss_no_lamb = - p_i_j * (torch.log(p_i_j) \
                              - torch.log(p_j) \
                              - torch.log(p_i))

    loss_no_lamb = loss_no_lamb.sum()

    return loss, loss_no_lamb

def IID_trace_plus_loss(plain_output, transformed_output, joint_probabilities_estimations, lamb=1.0, EPS=sys.float_info.epsilon):
    # has had softmax applied
    _, k = plain_output.size()
    joint_probability_matrix = compute_joint_with_estimations(plain_output, transformed_output,
                                                              joint_probabilities_estimations)
    assert (joint_probability_matrix.size() == (k, k))

    marginal_i = joint_probability_matrix.sum(dim=1).view(k, 1).expand(k, k)
    marginal_j = joint_probability_matrix.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    joint_probability_matrix[(joint_probability_matrix < EPS).data] = EPS
    marginal_j[(marginal_j < EPS).data] = EPS
    marginal_i[(marginal_i < EPS).data] = EPS

    # print(p_i_j.cpu().deach().numpy())
    loss = - joint_probability_matrix * (torch.log(joint_probability_matrix) \
                      - lamb * torch.log(marginal_j) \
                      - lamb * torch.log(marginal_i))

    loss = loss.sum()

    loss = loss + 0.5 * (1 - joint_probability_matrix.trace())

    loss_no_lamb = - joint_probability_matrix * (torch.log(joint_probability_matrix) \
                              - torch.log(marginal_j) \
                              - torch.log(marginal_i))

    loss_no_lamb = loss_no_lamb.sum()

    return loss, loss_no_lamb


def compute_linear_annealing(start_value, end_value, current_epoch, annealing_epochs):
    delta = (end_value - start_value) / annealing_epochs
    current_value = start_value + delta * current_epoch

    # Determines where to clip
    if end_value <= start_value:
        current_value = max(current_value, end_value)
    else:
        current_value = min(current_value, end_value)
    return current_value


def compute_loss_exponents(config, current_epoch):
    current_alpha = compute_linear_annealing(config.start_alpha, config.end_alpha, current_epoch,
                                             config.loss_anneal_epochs)
    current_beta = compute_linear_annealing(config.start_beta, config.end_beta, current_epoch,
                                            config.loss_anneal_epochs)

    return current_alpha, current_beta


def IID_entropy_trace_loss(x_out, x_tf_out, cluster_k, config, current_epoch, EPS=sys.float_info.epsilon):
    # has had softmax applied
    _, k = x_out.size()

    label_smoothing_loss = torch.tensor([0.0]).cuda();
    if config.label_smoothing_factor > 0:
        label_smoothing_loss = torch.mean(
            torch.cat((torch.sum(torch.log(x_out), dim=1), torch.sum(torch.log(x_tf_out), dim=1)), dim=0))
        label_smcompute_loss_exponentsoothing_loss = -label_smoothing_loss / cluster_k
        label_smoothing_loss *= config.label_smoothing_factor

    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1)

    p_i_j[(p_i_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    p_i_hentropy = -(p_i * torch.log(p_i)).sum()

    current_alpha, current_beta = compute_loss_exponents(config, current_epoch)

    # We want to maximize the quantity so we negate it
    loss = -(torch.pow(p_i_hentropy, current_alpha) * torch.pow(p_i_j.trace(), current_beta))
    loss = (1 - config.label_smoothing_factor) * loss + (config.label_smoothing_factor * label_smoothing_loss)

    return loss, p_i_j.trace(), label_smoothing_loss

def IID_entropy_trace_plus_loss(plain_outputs, transformed_outputs, joint_probabilities_estimations, cluster_k, config, current_epoch, EPS=sys.float_info.epsilon):
    # has had softmax applied
    _, k = plain_outputs.size()

    label_smoothing_loss = torch.tensor([0.0]).cuda();
    if config.label_smoothing_factor > 0:
        label_smoothing_loss = torch.mean(
            torch.cat((torch.sum(torch.log(plain_outputs), dim=1), torch.sum(torch.log(transformed_outputs), dim=1)), dim=0))
        label_smoothing_loss = -label_smoothing_loss / cluster_k
        label_smoothing_loss *= config.label_smoothing_factor

    joint_probability_matrix = compute_joint_with_estimations(plain_outputs, transformed_outputs, joint_probabilities_estimations)
    assert (joint_probability_matrix.size() == (k, k))

    marginal_i = joint_probability_matrix.sum(dim=1)

    joint_probability_matrix[(joint_probability_matrix < EPS).data] = EPS
    marginal_i[(marginal_i < EPS).data] = EPS

    p_i_hentropy = -(marginal_i * torch.log(marginal_i)).sum()

    current_alpha, current_beta = compute_loss_exponents(config, current_epoch)

    # We want to maximize the quantity so we negate it
    loss = -(torch.pow(p_i_hentropy, current_alpha) * torch.pow(joint_probability_matrix.trace(), current_beta))
    loss = (1 - config.label_smoothing_factor) * loss + (config.label_smoothing_factor * label_smoothing_loss)

    return loss, joint_probability_matrix.trace(), label_smoothing_loss

def IID_entropy_trace_plus_alpha(plain_output, transformed_output, old_joint_probabilities, alpha, config, current_epoch, EPS=sys.float_info.epsilon):
    # has had softmax applied

    joint_probability_matrix = compute_joint(plain_output, transformed_output)
    k = joint_probability_matrix.size()[0]

    # If the batch is not the first use an exponential average to estimate the probability matrix
    if old_joint_probabilities is not None:
        joint_probability_matrix = (1.0 - alpha) * joint_probability_matrix + alpha * old_joint_probabilities

    assert(joint_probability_matrix.size() == (k, k))

    marginal_i = joint_probability_matrix.sum(dim=1)

    joint_probability_matrix[(joint_probability_matrix < EPS).data] = EPS
    marginal_i[(marginal_i < EPS).data] = EPS

    p_i_hentropy = -(marginal_i * torch.log(marginal_i)).sum()

    current_alpha, current_beta = compute_loss_exponents(config, current_epoch)

    # We want to maximize the quantity so we negate it
    loss = -(torch.pow(p_i_hentropy, current_alpha) * torch.pow(joint_probability_matrix.trace(), current_beta))

    return loss, joint_probability_matrix.trace(), joint_probability_matrix.detach()

def IID_entropy_trace_plus_learnable_alpha(joint_probability_matrix, config, current_epoch, EPS=sys.float_info.epsilon):
    # has had softmax applied
    k = joint_probability_matrix.size()[0]

    assert(joint_probability_matrix.size() == (k, k))

    marginal_i = joint_probability_matrix.sum(dim=1)

    joint_probability_matrix[(joint_probability_matrix < EPS).data] = EPS
    marginal_i[(marginal_i < EPS).data] = EPS

    p_i_hentropy = -(marginal_i * torch.log(marginal_i)).sum()

    current_alpha, current_beta = compute_loss_exponents(config, current_epoch)

    # We want to maximize the quantity so we negate it
    loss = -(torch.pow(p_i_hentropy, current_alpha) * torch.pow(joint_probability_matrix.trace(), current_beta))

    return loss, joint_probability_matrix.trace()

def compute_mse_bounds(matrix_size, batch_size, confidence, num_samples=1000000):
    hit_probability = 1 / matrix_size

    samples = np.random.binomial(batch_size, hit_probability, num_samples)
    samples.sort()

    lower = samples[int(num_samples * confidence / 2)]
    upper = samples[int(-num_samples * confidence / 2)]
    return lower, upper


def IID_entropy_mse_loss(x_out, x_tf_out, cluster_k, lower_prob, upper_prob, config, EPS=sys.float_info.epsilon):
    # has had softmax applied
    _, k = x_out.size()

    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1)

    p_i_j[(p_i_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    target_joint = torch.zeros_like(p_i_j)
    matrix_size = p_i_j.size()[0]

    clamped_joint = torch.clamp(p_i_j, lower_prob, upper_prob).detach()
    for i in range(matrix_size):
        target_joint[i][i] = clamped_joint[i][i]

    mse_loss = torch.nn.MSELoss(reduction='sum')  # We sum each quadratic error
    final_loss = mse_loss(target_joint.view(-1, 1), p_i_j.view(-1, 1))

    return final_loss, final_loss

def logit_alignment_loss(all_logits):
    '''
    Computes the logit alignment loss term
    :param all_logits: list of network logit outputs of shape (batch_size, num_outputs)
                       each tensor must have the same number of output
    :return: list of loss terms, one for each pair of logits in the input sequence
    '''
    mse_loss = nn.MSELoss()
    demeaned_logits = [None] * len(all_logits)

    # Demeans all logits individually
    for i in range(len(all_logits)):
        demeaned_logits[i] = all_logits[i] - all_logits[i].mean(dim=1, keepdim=True)

    all_loss_pairs = []

    # foreach pair of logits
    for first in range(len(demeaned_logits)):
        for second in range(first + 1, len(demeaned_logits)):
            first_logits = demeaned_logits[first]
            second_logits = demeaned_logits[second]

            current_pair_loss = mse_loss(first_logits, second_logits)
            all_loss_pairs.append(current_pair_loss)

    return all_loss_pairs


if __name__ == "__main__":
    compute_mse_bounds(350, 185, 0.15)
