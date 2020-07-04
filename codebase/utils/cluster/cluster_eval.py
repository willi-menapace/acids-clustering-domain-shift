from __future__ import print_function

import itertools
import sys
from datetime import datetime
import os
import shutil

import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import mutual_info_score
import scipy
import scipy.stats
import torch
from matplotlib import pyplot as plt

from PIL import Image

from .IID_losses import IID_loss
from .eval_metrics import _hungarian_match, _original_match, _acc
from .transforms import sobel_process

from codebase.utils.cluster.train_utils import build_train_batch, build_eval_batch


def save_helper(net, config, images, images_cropped, batch, head_a_output_dir, head_b_output_dir, image_id_start,
                prefix):
    # If the network needs cropped images pass also the cropped ones
    if config.crop_network:
        predictions_a = net(images, images_cropped, head="A")[0]
        predictions_b = net(images, images_cropped, head="B")[0]
    else:
        predictions_a = net(images, head="A")[0]
        predictions_b = net(images, head="B")[0]

    # Passes images though the ground thruth head
    cluster_indexes_a = torch.argmax(predictions_a, dim=1)
    cluster_indexes_b = torch.argmax(predictions_b, dim=1)

    for idx, (image, original_image, cluster_index_a, cluster_index_b) in enumerate(
            zip(images, batch, cluster_indexes_a, cluster_indexes_b)):
        original_image = original_image[0]

        cluster_a_score = predictions_a[idx][cluster_index_a]
        cluster_b_score = predictions_b[idx][cluster_index_b]
        image_dir_a = os.path.join(head_a_output_dir, str(cluster_index_a.cpu().data.numpy()))
        image_dir_b = os.path.join(head_b_output_dir, str(cluster_index_b.cpu().data.numpy()))

        np_image = image.mean(dim=0, keepdim=True).cpu().data.numpy()
        minimum = np_image.min()
        maximum = np_image.max()
        np_image = ((np_image - minimum) / (maximum - minimum)) * 255.0
        pil_image = Image.fromarray(np.rollaxis(np.concatenate([np_image] * 3).astype(np.uint8), 0, 3))
        pil_image.save(os.path.join(image_dir_a, "{}_{}_{:.3f}.png".format(prefix, image_id_start, cluster_a_score)))
        original_image.save(
            os.path.join(image_dir_a, "{}_orig_{}_{:.3f}.png".format(prefix, image_id_start, cluster_a_score)))
        pil_image.save(os.path.join(image_dir_b, "{}_{}_{:.3f}.png".format(prefix, image_id_start, cluster_b_score)))
        original_image.save(
            os.path.join(image_dir_b, "{}_orig_{}_{:.3f}.png".format(prefix, image_id_start, cluster_b_score)))

        # If the image is cropped save also the cropped version
        if config.crop_network:
            np_image_cropped = images_cropped[idx].mean(dim=0, keepdim=True).cpu().data.numpy()
            minimum = np_image_cropped.min()
            maximum = np_image_cropped.max()
            np_image_cropped = ((np_image_cropped - minimum) / (maximum - minimum)) * 255.0
            pil_image_cropped = Image.fromarray(
                np.rollaxis(np.concatenate([np_image_cropped] * 3).astype(np.uint8), 0, 3))
            pil_image_cropped.save(
                os.path.join(image_dir_a, "{}_{}_cropped_{:.3f}.png".format(prefix, image_id_start, cluster_a_score)))
            pil_image_cropped.save(
                os.path.join(image_dir_b, "{}_{}_cropped_{:.3f}.png".format(prefix, image_id_start, cluster_b_score)))

        image_id_start += 1


def save_k_image_batches(config, output_dir, net, dataloader, tf1, tf2, tf3, crop_transform, preprocessing_pool,
                         sobel=False, k=4, verbose=0):
    '''
    Saves the first k batches of images for both head A and B under the output_dir directory
    '''

    net.eval()

    with torch.no_grad():

        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)

        head_a_output_dir = os.path.join(output_dir, "a")
        head_b_output_dir = os.path.join(output_dir, "b")
        os.mkdir(head_a_output_dir)
        os.mkdir(head_b_output_dir)

        for cluster_idx in range(config.output_k_A):
            cluster_dir_path = os.path.join(head_a_output_dir, str(cluster_idx))
            os.mkdir(cluster_dir_path)

        for cluster_idx in range(config.output_k_B):
            cluster_dir_path = os.path.join(head_b_output_dir, str(cluster_idx))
            os.mkdir(cluster_dir_path)

        current_id = 0
        for b_i, batch in enumerate(dataloader):
            if b_i >= k:
                break

            images, images_cropped = build_eval_batch(batch, config, tf3, crop_transform, preprocessing_pool, sobel)
            save_helper(net, config, images, images_cropped, batch, head_a_output_dir, head_b_output_dir, current_id,
                        "tf3")

            results = build_train_batch(batch, config, tf1, tf2, crop_transform, preprocessing_pool)
            images, images_cropped = results["plain"], results["plain_cropped"]
            save_helper(net, config, images, images_cropped, batch, head_a_output_dir, head_b_output_dir, current_id,
                        "tf1")

            images, images_cropped = results["transformed"], results["transformed_cropped"]
            save_helper(net, config, images, images_cropped, batch, head_a_output_dir, head_b_output_dir, current_id,
                        "tf2")

            current_id += len(batch)

    net.train()


def _clustering_get_data(config, net, dataloader, tf3, crop_transform, preprocessing_pool, sobel=False,
                         using_IR=False, get_soft=False, verbose=0):
    """
    Returns cuda tensors for flat preds and targets.
    """

    assert (not using_IR)  # sanity; IR used by segmentation only

    num_batches = len(dataloader)
    flat_targets_all = torch.zeros((num_batches * config.batch_sz),
                                   dtype=torch.int32).cuda()
    flat_predss_all = [torch.zeros((num_batches * config.batch_sz),
                                   dtype=torch.int32).cuda() for _ in
                       range(config.num_sub_heads)]

    if get_soft:
        soft_predss_all = [torch.zeros((num_batches * config.batch_sz,
                                        config.output_k),
                                       dtype=torch.float32).cuda() for _ in range(
            config.num_sub_heads)]

    num_test = 0
    for b_i, batch in enumerate(dataloader):

        images, images_cropped = build_eval_batch(batch, config, tf3, crop_transform, preprocessing_pool, sobel)

        # If the batch is already in the form of tensors extract it directly, else create the tensor with the labels
        if not isinstance(batch, list):
            flat_targets = batch[1]
        else:
            flat_targets = torch.IntTensor([element[1] for element in batch])

        with torch.no_grad():
            # If the network needs cropped images pass also the cropped ones
            if config.crop_network:
                x_outs = net(images, images_cropped, ground_truth=flat_targets)
            else:
                x_outs = net(images, ground_truth=flat_targets)

        # assert (x_outs[0].shape[1] == config.output_k)
        assert (len(x_outs[0].shape) == 2)

        num_test_curr = flat_targets.shape[0]
        num_test += num_test_curr

        start_i = b_i * config.batch_sz
        for i in range(config.num_sub_heads):
            x_outs_curr = x_outs[i]
            flat_preds_curr = torch.argmax(x_outs_curr, dim=1)  # along output_k
            flat_predss_all[i][start_i:(start_i + num_test_curr)] = flat_preds_curr

            if get_soft:
                soft_predss_all[i][start_i:(start_i + num_test_curr), :] = x_outs_curr

        flat_targets_all[start_i:(start_i + num_test_curr)] = flat_targets.cuda()

    flat_predss_all = [flat_predss_all[i][:num_test] for i in
                       range(config.num_sub_heads)]
    flat_targets_all = flat_targets_all[:num_test]

    if not get_soft:
        return flat_predss_all, flat_targets_all
    else:
        soft_predss_all = [soft_predss_all[i][:num_test] for i in
                           range(config.num_sub_heads)]

        return flat_predss_all, flat_targets_all, soft_predss_all


def cluster_subheads_eval(config, net,
                          mapping_assignment_dataloader,
                          mapping_test_dataloader,
                          sobel,
                          using_IR=False,
                          get_data_fn=_clustering_get_data,
                          use_sub_head=None,
                          verbose=0):
    """
    Used by both clustering and segmentation.
    Returns metrics for test set.
    Get result from average accuracy of all sub_heads (mean and std).
    All matches are made from training data.
    Best head metric, which is order selective unlike mean/std, is taken from
    best head determined by training data (but metric computed on test data).

    ^ detail only matters for IID+/semisup where there's a train/test split.

    Option to choose best sub_head either based on loss (set use_head in main
    script), or eval. Former does not use labels for the selection at all and this
    has negligible impact on accuracy metric for our models.
    """

    all_matches, train_accs = _get_assignment_data_matches(net,
                                                           mapping_assignment_dataloader,
                                                           config,
                                                           sobel=sobel,
                                                           using_IR=using_IR,
                                                           get_data_fn=get_data_fn,
                                                           verbose=verbose)

    best_sub_head_eval = np.argmax(train_accs)
    if (config.num_sub_heads > 1) and (use_sub_head is not None):
        best_sub_head = use_sub_head
    else:
        best_sub_head = best_sub_head_eval

    if config.mode == "IID":
        assert (
                config.mapping_assignment_partitions == config.mapping_test_partitions)
        test_accs = train_accs
    elif config.mode == "IID+":
        flat_predss_all, flat_targets_all, = \
            get_data_fn(config, net, mapping_test_dataloader, sobel=sobel,
                        using_IR=using_IR,
                        verbose=verbose)

        num_samples = flat_targets_all.shape[0]
        test_accs = np.zeros(config.num_sub_heads, dtype=np.float32)
        for i in range(config.num_sub_heads):
            reordered_preds = torch.zeros(num_samples,
                                          dtype=flat_predss_all[0].dtype).cuda()
            for pred_i, target_i in all_matches[i]:
                reordered_preds[flat_predss_all[i] == pred_i] = target_i
            test_acc = _acc(reordered_preds, flat_targets_all, config.gt_k, verbose=0)

            test_accs[i] = test_acc
    else:
        assert (False)

    return {"test_accs": list(test_accs),
            "avg": np.mean(test_accs),
            "std": np.std(test_accs),
            "best": test_accs[best_sub_head],
            "worst": test_accs.min(),
            "best_train_sub_head": best_sub_head,  # from training data
            "best_train_sub_head_match": all_matches[best_sub_head],
            "train_accs": list(train_accs)}


def _get_assignment_data_matches(net, mapping_assignment_dataloader, config,
                                 sobel=False,
                                 using_IR=False,
                                 get_data_fn=None,
                                 just_matches=False,
                                 verbose=0):
    """
    Get all best matches per head based on train set i.e. mapping_assign,
    and mapping_assign accs.
    """

    if verbose:
        print("calling cluster eval direct (helper) %s" % datetime.now())
        sys.stdout.flush()

    flat_predss_all, flat_targets_all = \
        get_data_fn(config, net, mapping_assignment_dataloader, sobel=sobel,
                    using_IR=using_IR,
                    verbose=verbose)

    if verbose:
        print("getting data fn has completed %s" % datetime.now())
        print("flat_targets_all %s, flat_predss_all[0] %s" %
              (list(flat_targets_all.shape), list(flat_predss_all[0].shape)))
        sys.stdout.flush()

    num_test = flat_targets_all.shape[0]
    if verbose == 2:
        print("num_test: %d" % num_test)
        for c in range(config.gt_k):
            print("gt_k: %d count: %d" % (c, (flat_targets_all == c).sum()))

    assert (flat_predss_all[0].shape == flat_targets_all.shape)
    num_samples = flat_targets_all.shape[0]

    all_matches = []
    if not just_matches:
        all_accs = np.zeros(config.num_sub_heads, dtype=np.float32)

    for i in range(config.num_sub_heads):
        if verbose:
            print("starting head %d with eval mode %s, %s" % (i, config.eval_mode,
                                                              datetime.now()))
            sys.stdout.flush()

        if config.eval_mode == "hung":
            match = _hungarian_match(flat_predss_all[i], flat_targets_all,
                                     preds_k=config.output_k,
                                     targets_k=config.gt_k)
        elif config.eval_mode == "orig":
            match = _original_match(flat_predss_all[i], flat_targets_all,
                                    preds_k=config.output_k,
                                    targets_k=config.gt_k)
        else:
            assert (False)

        if verbose:
            print("got match %s" % (datetime.now()))
            sys.stdout.flush()

        all_matches.append(match)

        if not just_matches:
            # reorder predictions to be same cluster assignments as gt_k
            found = torch.zeros(config.output_k)
            reordered_preds = torch.zeros(num_samples,
                                          dtype=flat_predss_all[0].dtype).cuda()

            for pred_i, target_i in match:
                # reordered_preds[flat_predss_all[i] == pred_i] = target_i
                reordered_preds[torch.eq(flat_predss_all[i], int(pred_i))] = torch.from_numpy(
                    np.array(target_i)).cuda().int().item()
                found[pred_i] = 1
                if verbose == 2:
                    print((pred_i, target_i))
            assert (found.sum() == config.output_k)  # each output_k must get mapped

            if verbose:
                print("reordered %s" % (datetime.now()))
                sys.stdout.flush()

            acc = _acc(reordered_preds, flat_targets_all, config.gt_k, verbose)
            all_accs[i] = acc

    if just_matches:
        return all_matches
    else:
        return all_matches, all_accs


def get_subhead_using_loss(config, dataloaders_head_B, net, sobel, lamb,
                           compare=False):
    net.eval()

    head = "B"  # main output head
    dataloaders = dataloaders_head_B
    iterators = (d for d in dataloaders)

    b_i = 0
    loss_per_sub_head = np.zeros(config.num_sub_heads)
    for tup in itertools.izip(*iterators):
        net.module.zero_grad()

        dim = config.in_channels
        if sobel:
            dim -= 1

        all_imgs = torch.zeros(config.batch_sz, dim,
                               config.input_sz,
                               config.input_sz).cuda()
        all_imgs_tf = torch.zeros(config.batch_sz, dim,
                                  config.input_sz,
                                  config.input_sz).cuda()

        imgs_curr = tup[0][0]  # always the first
        curr_batch_sz = imgs_curr.size(0)
        for d_i in range(config.num_dataloaders):
            imgs_tf_curr = tup[1 + d_i][0]  # from 2nd to last
            assert (curr_batch_sz == imgs_tf_curr.size(0))

            actual_batch_start = d_i * curr_batch_sz
            actual_batch_end = actual_batch_start + curr_batch_sz
            all_imgs[actual_batch_start:actual_batch_end, :, :, :] = \
                imgs_curr.cuda()
            all_imgs_tf[actual_batch_start:actual_batch_end, :, :, :] = \
                imgs_tf_curr.cuda()

        curr_total_batch_sz = curr_batch_sz * config.num_dataloaders
        all_imgs = all_imgs[:curr_total_batch_sz, :, :, :]
        all_imgs_tf = all_imgs_tf[:curr_total_batch_sz, :, :, :]

        if sobel:
            all_imgs = sobel_process(all_imgs, config.include_rgb)
            all_imgs_tf = sobel_process(all_imgs_tf, config.include_rgb)

        with torch.no_grad():
            x_outs = net(all_imgs, head=head)
            x_tf_outs = net(all_imgs_tf, head=head)

        for i in range(config.num_sub_heads):
            loss, loss_no_lamb = IID_loss(x_outs[i], x_tf_outs[i],
                                          lamb=lamb)
            loss_per_sub_head[i] += loss.item()

        if b_i % 100 == 0:
            print("at batch %d" % b_i)
            sys.stdout.flush()
        b_i += 1

    best_sub_head_loss = np.argmin(loss_per_sub_head)

    if compare:
        print(loss_per_sub_head)
        print("best sub_head by loss: %d" % best_sub_head_loss)

        best_epoch = np.argmax(np.array(config.epoch_acc))
        if "best_train_sub_head" in config.epoch_stats[best_epoch]:
            best_sub_head_eval = config.epoch_stats[best_epoch]["best_train_sub_head"]
            test_accs = config.epoch_stats[best_epoch]["test_accs"]
        else:  # older config version
            best_sub_head_eval = config.epoch_stats[best_epoch]["best_head"]
            test_accs = config.epoch_stats[best_epoch]["all"]

        print("best sub_head by eval: %d" % best_sub_head_eval)

        print("... loss select acc: %f, eval select acc: %f" %
              (test_accs[best_sub_head_loss],
               test_accs[best_sub_head_eval]))

    net.train()

    return best_sub_head_loss


def compute_cluster_confusion_matrix(clusters, ground_truth_clusters, cluster_k, ground_truth_k):
    '''
    Computes the confusion matrix of clusters vs cluster assignments
    :param clusters: numpy vector representing cluster assignments
    :param ground_truth_clusters: numpy vector representing ground truth
    :param cluster_k: number of clusters
    :param ground_truth_k: number of classes in the ground truth
    :return: a (ground_truth_k, ground_truth_k) confusion matrix
    '''

    confusion_matrix = np.zeros((cluster_k, ground_truth_k))

    for current_cluster, current_ground_truth_cluster in zip(clusters, ground_truth_clusters):
        confusion_matrix[current_cluster, current_ground_truth_cluster] += 1

    return confusion_matrix


def cluster_eval(config, net, test_dataloader, tf3, crop_transform, preprocessing_pool, sobel):
    net.eval()

    # Computed predicted clusters and gets ground truth
    predicted_clusters, ground_truth_clusters = _clustering_get_data(config, net, test_dataloader, tf3, crop_transform,
                                                                     preprocessing_pool, sobel=sobel, using_IR=False,
                                                                     verbose=False)
    predicted_clusters = predicted_clusters[0]
    num_samples = predicted_clusters.shape[0]

    # Computes accuracy if the number of predicted clusters matches the number of ground truth ones
    accuracy = None
    if config.gt_k == config.output_k_B:
        match = _hungarian_match(predicted_clusters, ground_truth_clusters, config.gt_k, config.output_k_B)

        found = torch.zeros(config.gt_k)
        reordered_preds = torch.zeros(num_samples, dtype=predicted_clusters.dtype).cuda()

        for pred_i, target_i in match:
            # reordered_preds[flat_predss_all[i] == pred_i] = target_i
            reordered_preds[torch.eq(predicted_clusters, int(pred_i))] = torch.from_numpy(
                np.array(target_i)).cuda().int().item()
            found[pred_i] = 1
        assert (found.sum() == config.gt_k)  # each output_k must get mapped

        accuracy = int((reordered_preds == ground_truth_clusters).sum()) / float(num_samples)

    predicted_clusters = predicted_clusters.cpu().numpy()
    ground_truth_clusters = ground_truth_clusters.cpu().numpy()

    confusion_matrix = compute_cluster_confusion_matrix(predicted_clusters, ground_truth_clusters, config.output_k_B,
                                                        config.gt_k)

    # Computes entropies
    _, predicted_clusters_distribution = np.unique(predicted_clusters, return_counts=True)
    predicted_clusters_entropy = scipy.stats.entropy(predicted_clusters_distribution)

    _, ground_truth_clusters_distribution = np.unique(ground_truth_clusters, return_counts=True)
    ground_truth_clusters_entropy = scipy.stats.entropy(ground_truth_clusters_distribution)

    # Computes information scores
    mutual_information = mutual_info_score(predicted_clusters, ground_truth_clusters)
    conditional_entropy = -(mutual_information - ground_truth_clusters_entropy)
    nmi = normalized_mutual_info_score(predicted_clusters, ground_truth_clusters)

    net.train()

    return nmi, mutual_information, conditional_entropy, ground_truth_clusters_entropy, predicted_clusters_entropy, accuracy, confusion_matrix


def make_confusion_matrix_plot(confusion_matrix, save_location, prefix=""):
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)

    ax.set_title("Class confusion matrix")
    fig.tight_layout()
    plt.savefig(os.path.join(save_location, "{}confusion_matrix.pdf".format(prefix)))

    normalized_matrix = np.zeros(confusion_matrix.shape)

    for i in range(confusion_matrix.shape[0]):
        sum = confusion_matrix[i].sum()
        if sum > 0:
            normalized_matrix[i] = confusion_matrix[i] / sum

    fig, ax = plt.subplots()
    im = ax.imshow(normalized_matrix)

    ax.set_title("Normalized confusion matrix")
    fig.tight_layout()
    plt.savefig(os.path.join(save_location, "{}normalized_confusion_matrix.pdf".format(prefix)))

    transposed_confusion_matrix = np.transpose(confusion_matrix)
    normalized_transposed_matrix = np.zeros(transposed_confusion_matrix.shape)

    for i in range(transposed_confusion_matrix.shape[0]):
        sum = transposed_confusion_matrix[i].sum()
        if sum > 0:
            normalized_transposed_matrix[i] = transposed_confusion_matrix[i] / sum

    fig, ax = plt.subplots()
    im = ax.imshow(normalized_transposed_matrix)

    ax.set_title("Normalized transposed confusion matrix")
    fig.tight_layout()
    plt.savefig(os.path.join(save_location, "{}transposed_normalized_confusion_matrix.pdf".format(prefix)))
