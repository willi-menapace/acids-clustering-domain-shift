from __future__ import print_function

import argparse
import json
import os
import pickle

from datetime import datetime
import multiprocessing as mp

import matplotlib
import torch

from codebase.utils.cluster.dict_wrapper import DictWrapper
from codebase.utils.cluster.management.basic_dial_evaluation_manager import BasicDialEvaluationManager
from codebase.utils.cluster.management.basic_dial_optimizer_manager import BasicDialOptimizerManager
from codebase.utils.cluster.management.basic_dial_state_manager import BasicDialStateManager
from codebase.utils.cluster.management.mixed_domain_batching import MixedDomainTrainBatchLoader
from codebase.utils.cluster.management.mixed_domain_network_manager import MixedDomainNetworkManager
from codebase.utils.cluster.management.statistics_manager import StatisticsManager
from codebase.utils.cluster.train_utils import create_directories, check_and_setup_basic_dial_config, perform_evaluation

matplotlib.use('Agg')

# Loads configuration file
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
arguments = parser.parse_args()

config = None
original_config = None
with open(arguments.config) as json_file:
    config = json.load(json_file)
    original_config = config
    config = DictWrapper(config)

# Check if the configuration is valid and enriches it with additional computed fields
check_and_setup_basic_dial_config(config)

# Creates all the directories necessary to save results
create_directories(config)

print("Starting preprocessing pool")
preprocessing_pool = mp.Pool(8)

# Creates the training dataloader
train_batch_loader_map = {'A': MixedDomainTrainBatchLoader(config, preprocessing_pool), 'B': MixedDomainTrainBatchLoader(config, preprocessing_pool)}
network_manager = MixedDomainNetworkManager(config)
network_manager.to_parallel_gpu()
network_manager.train()
optimizer_manager = BasicDialOptimizerManager(config, network_manager)
state_manager = BasicDialStateManager()

evaluation_manager = BasicDialEvaluationManager(config, network_manager, preprocessing_pool)

statistics_manager = StatisticsManager(config)
evaluation_statistics_manager = StatisticsManager(config, writer=statistics_manager.writer)

# Average number of steps needed to complete an epoch across all domains
steps_per_epoch = train_batch_loader_map['A'].get_avg_steps_per_epoch()

start_step = 0

# Loads training state
if config.restart:
    checkpoint_name = os.path.join(config.out_dir, "latest.pth.tar")
    if os.path.isfile(checkpoint_name):
        print("Loading checkpoint: {}".format(checkpoint_name))

        start_step = state_manager.load(checkpoint_name, network_manager, optimizer_manager)

    else:
        config.restart = False
        print("Warning: loading checkpoint requested, but no checkpoint found at {}".format(checkpoint_name))

heads = ["B", "A"]
if config.head_A_first:
    heads = ["A", "B"]

head_epochs = {}
head_epochs["A"] = config.head_A_epochs
head_epochs["B"] = config.head_B_epochs

current_step = start_step

# Train ------------------------------------------------------------------------------

while current_step < config.num_steps:

    # An epoch is defined as a complete pass over the dataset of whatever head
    current_epoch = int(current_step / steps_per_epoch)
    print("")
    print("---------- Epoch [{}] Step [{}] ----------".format(current_epoch, current_step))

    for head_i in range(2):

        head = heads[head_i]
        if config.skip_a and head == 'A':
            continue

        for batch_index in range(steps_per_epoch):

            # Gets batch
            train_batch_loader = train_batch_loader_map[head]
            training_batch = train_batch_loader.get_next_batch()

            # Computes losses for the current batch
            main_cluster_loss, domain_cluster_loss, logit_variance_loss, _, per_domain_cluster_losses, centroid_alignment_loss = network_manager.compute_cluster_losses(training_batch, head, current_epoch)
            average_per_domain_cluster_loss = torch.mean(per_domain_cluster_losses)

            # Anneal the centroid loss on the first epochs
            if current_epoch < config.centroid_estimation_annealing_epochs:
                centroid_alignment_loss = centroid_alignment_loss * current_epoch / config.centroid_estimation_annealing_epochs

            domain_cluster_annealing_factor = 1
            if current_epoch < config.domain_cluster_lambda_annealing_epochs:
                domain_cluster_annealing_factor = current_epoch / config.domain_cluster_lambda_annealing_epochs

            total_loss = main_cluster_loss + config.domain_cluster_lambda * domain_cluster_annealing_factor * domain_cluster_loss + config.logit_variance_lambda * logit_variance_loss + config.domain_specific_lambda * average_per_domain_cluster_loss + config.centroid_alignment_lambda * centroid_alignment_loss

            # We make a step every backward_steps backwards, so zero the gradient at the beginning of such cycle
            if batch_index % config.backward_steps == 0:
                optimizer_manager.encoder_optimizer.zero_grad()
            total_loss.backward()

            # We make a step every backward_steps backwards, so perform a training step at the end of such cycle or end of epoch
            if ((batch_index % config.backward_steps) == (config.backward_steps - 1)) or batch_index == steps_per_epoch - 1:
                optimizer_manager.encoder_optimizer.step()

            current_step += 1

            statistics_manager.update('clustering_head_{}/loss'.format(head), total_loss.item())
            statistics_manager.update('clustering_head_{}/main_cluster_loss'.format(head), main_cluster_loss.item())
            statistics_manager.update('clustering_head_{}/domain_cluster_loss'.format(head), domain_cluster_loss.item())
            statistics_manager.update('clustering_head_{}/logit_loss'.format(head), logit_variance_loss.item())
            statistics_manager.update('clustering_head_{}/centroid_alignment_loss'.format(head), centroid_alignment_loss.item())


            for domain_idx in range(config.domains_count):
                statistics_manager.update('clustering_head_{}/domain_{}_cluster_loss'.format(head, domain_idx), per_domain_cluster_losses[domain_idx].item())

            print("  [{}/{}] Head {}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {}".format(batch_index, steps_per_epoch, head, total_loss.item(), main_cluster_loss.item(), domain_cluster_loss.item(), logit_variance_loss.item(), average_per_domain_cluster_loss, centroid_alignment_loss.item(), datetime.now()))

    # Flushes statistics to tensorboard after each epoch
    statistics_manager.write_to_tensorboard(current_step)

    # Performs evaluation every given number of epochs
    if current_epoch % config.evaluation_epochs == 0:

        if config.test_domains_count > 0 and config.double_eval == False:
            raise Exception("Using test domains without double evaluation would result in no estimation for their batch norm parameters."
                            "Enable double evaluation or create ad hoc code for parameter estimation")
        if config.double_eval:

            skip_test_domains = False
            # If the user requests test domains to be tested using the DA layers of the training domains,
            # then we do not perform statistics estimation for them
            if len(config.map_test_to_domains) > 0:
                skip_test_domains = True

            # Performs evaluation staying in evaluation mode to update batch norm parameters
            perform_evaluation(config, evaluation_manager, evaluation_statistics_manager, current_epoch, current_step, visualize=False, skip_test_domains=skip_test_domains, prefix="double_")

        network_manager.eval()
        perform_evaluation(config, evaluation_manager, evaluation_statistics_manager, current_epoch, current_step, visualize=(current_epoch % config.visualization_epochs == 0 and current_epoch > 0), prefix="")
        network_manager.train()
        evaluation_statistics_manager.write_to_tensorboard(current_step)

    # Saves at regular intervals
    if current_epoch % config.save_freq == 0:

        # Saves running checkpoint
        checkpoint_name = os.path.join(config.out_dir, "latest.pth.tar")
        state_manager.save(checkpoint_name, current_step, network_manager, optimizer_manager)

        # Every 10 epochs also save permanent checkpoint
        if current_epoch % 10 == 0:
            checkpoint_name = os.path.join(config.checkpoint_directory, "{}.pth.tar".format(current_step))
            state_manager.save(checkpoint_name, current_step, network_manager, optimizer_manager)

    with open(os.path.join(config.out_dir, "config.pickle"), 'wb') as outfile:
        pickle.dump(dict(original_config), outfile)

    with open(os.path.join(config.out_dir, "config.txt"), "w") as text_file:
        text_file.write("%s" % dict(original_config))

preprocessing_pool.close()



