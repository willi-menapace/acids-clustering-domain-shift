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
from codebase.utils.cluster.management.generalization_refinement_batching import RefinementTrainBatchLoader
from codebase.utils.cluster.management.mixed_domain_network_manager import MixedDomainNetworkManager
from codebase.utils.cluster.management.statistics_manager import StatisticsManager
from codebase.utils.cluster.train_utils import create_directories, perform_evaluation, check_and_setup_generalization_refinement_config

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
check_and_setup_generalization_refinement_config(config)

# Creates all the directories necessary to save results
create_directories(config)

print("Starting preprocessing pool")
preprocessing_pool = mp.Pool(8)


train_batch_loader_map = {'A': RefinementTrainBatchLoader(config, preprocessing_pool), 'B': RefinementTrainBatchLoader(config, preprocessing_pool)}
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
    start_checkpoint_name = os.path.join(config.out_dir, "start.pth.tar")
    checkpoint_name = os.path.join(config.out_dir, "latest.pth.tar")
    if os.path.isfile(checkpoint_name):
        print("Loading checkpoint: {}".format(checkpoint_name))
        start_step = state_manager.load(checkpoint_name, network_manager, optimizer_manager)
    elif os.path.isfile(start_checkpoint_name):
        print("Loading start checkpoint: {}".format(start_checkpoint_name))
        start_step = state_manager.load(start_checkpoint_name, network_manager, optimizer_manager, discard_optimizer=False)

    else:
        config.restart = False
        print("Warning: loading checkpoint requested, but no checkpoint found at {}".format(checkpoint_name))
        raise Exception("Refinement cannot proceed without a valid starting checkpoint")
else:
    raise Exception("No restart was requested, but refinement requires starting from a given checkpoint.")

heads = ["B", "A"]
if config.head_A_first:
    heads = ["A", "B"]

head_epochs = {}
head_epochs["A"] = config.head_A_epochs
head_epochs["B"] = config.head_B_epochs

current_step = start_step
current_epoch = int(current_step / steps_per_epoch)

# Perform initial evaluation
network_manager.eval()
perform_evaluation(config, evaluation_manager, evaluation_statistics_manager, current_epoch, current_step, visualize=(current_epoch % config.visualization_epochs == 0), prefix="")
network_manager.train()
evaluation_statistics_manager.write_to_tensorboard(current_step)


# Train ------------------------------------------------------------------------------
while current_step < config.num_steps:

    # An epoch is defined as a complete pass over the dataset of whatever head
    current_epoch = int(current_step / steps_per_epoch)

    # Makes it divisible by 2 for compatibility
    if current_epoch % 2 != 0:
        current_epoch -= 1

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

            if config.refinement_method == "IIC":
                # Computes losses for the current batch
                main_refinement_loss, logit_variance_loss, _ = network_manager.compute_refinement_losses(training_batch, head)
            elif config.refinement_method == "fixmatch":


                # Computes fixmatch annealing
                use_fixmatch = False
                fixmatch_threshold = 1.0
                start = config.fixmatch_threshold_anneal_start
                end = config.fixmatch_threshold_anneal_end
                if current_step > start:
                    use_fixmatch = True
                    fixmatch_threshold = max(1.0 + (config.fixmatch_threshold - 1.0) / (end - start) * (current_step - start), config.fixmatch_threshold)

                # Computes losses for the current batch
                main_refinement_loss, logit_variance_loss, _ = network_manager.compute_refinement_losses(training_batch, head, use_fixmatch=use_fixmatch, fixmatch_threshold=fixmatch_threshold)
            elif config.refinement_method == "entropy":
                main_refinement_loss, logit_variance_loss, _ = network_manager.compute_refinement_entropy_losses(training_batch, head, config.refinement_entropy_threshold)
            else:
                raise NotImplementedError("Refimenent method '' is not implemented".format(config.refinement_method))

            total_loss = main_refinement_loss + config.logit_variance_lambda * logit_variance_loss

            # We make a step every backward_steps backwards, so zero the gradient at the beginning of such cycle
            if batch_index % config.backward_steps == 0:
                optimizer_manager.encoder_optimizer.zero_grad()
            total_loss.backward()

            # We make a step every backward_steps backwards, so perform a training step at the end of such cycle or end of epoch
            if ((batch_index % config.backward_steps) == (config.backward_steps - 1)) or batch_index == steps_per_epoch - 1:
                optimizer_manager.encoder_optimizer.step()

            current_step += 1

            statistics_manager.update('clustering_head_{}/loss'.format(head), total_loss.item())
            statistics_manager.update('clustering_head_{}/main_cluster_loss'.format(head), main_refinement_loss.item())
            statistics_manager.update('clustering_head_{}/logit_loss'.format(head), logit_variance_loss.item())


            print("  [{}/{}] Head {}, {:.3f}, {:.3f}, {:.3f}, {}".format(batch_index, steps_per_epoch, head, total_loss.item(), main_refinement_loss.item(), logit_variance_loss.item(), datetime.now()))

    # Flushes statistics to tensorboard after each epoch
    statistics_manager.write_to_tensorboard(current_step)

    # Evaluates the test domain to estimate bn parameters
    if config.double_eval:
        with torch.no_grad():
            assert(config.test_domains_count == 1) # Finetuning allows only one test domain
            evaluation_manager.evaluate(config.domains_count)

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
        perform_evaluation(config, evaluation_manager, evaluation_statistics_manager, current_epoch, current_step, visualize=(current_epoch % config.visualization_epochs == 0), prefix="")
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



