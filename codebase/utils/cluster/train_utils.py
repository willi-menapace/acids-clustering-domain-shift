import os
from statistics import mean

import torch

from codebase.utils.cluster.data import transform_list
from codebase.utils.cluster.dict_wrapper import DictWrapper
from codebase.utils.cluster.transforms import sobel_process, custom_cutout


def build_eval_batch(current_batch, config, tf3, crop_transform, preprocessing_pool, sobel=True):
    current_eval_batch = transform_list(current_batch, tf3, preprocessing_pool).cuda()
    if sobel:
        current_eval_batch = sobel_process(current_eval_batch, config.include_rgb)

    current_eval_batch_cropped = None
    if config.crop_network:
        current_batch_cropped = transform_list(current_batch, crop_transform, preprocessing_pool)
        current_eval_batch_cropped = transform_list(current_batch_cropped, tf3, preprocessing_pool).cuda()
        if sobel:
            current_eval_batch_cropped = sobel_process(current_eval_batch_cropped, config.include_rgb)

    return current_eval_batch, current_eval_batch_cropped


def build_train_batch(current_batch, config, tf1, tf2, crop_transform, preprocessing_pool):
    '''
    Builds an IIC batch starting from a batch of plain images

    :param current_batch: The batch of plain images
    :param config: train configuration
    :param tf1: transformation to apply to obtain plain IIC images
    :param tf2: transformation to apply to obtain the transformed version of the correspoinding plain IIC image
    :param preprocessing_pool: pool object for parallel image transofrmation
    :return:
    '''
    # one less because this is before sobel
    plain_images_final_batch = torch.zeros(config.batch_sz, config.in_channels - 1,
                                           config.input_sz,
                                           config.input_sz).cuda()
    transformed_images_final_batch = torch.zeros(config.batch_sz, config.in_channels - 1,
                                                 config.input_sz,
                                                 config.input_sz).cuda()

    # In case of CropNet allocate space also for the crops
    if config.crop_network:
        plain_images_final_batch_cropped = torch.zeros(config.batch_sz, config.in_channels - 1,
                                                       config.input_sz,
                                                       config.input_sz).cuda()
        transformed_images_final_batch_cropped = torch.zeros(config.batch_sz, config.in_channels - 1,
                                                             config.input_sz,
                                                             config.input_sz).cuda()

        # Creates a cropped version of the plain images
        current_batch_cropped = transform_list(current_batch, crop_transform, preprocessing_pool)
        plain_images_batch_cropped = transform_list(current_batch_cropped, tf1, preprocessing_pool)

    plain_images_batch = transform_list(current_batch, tf1, preprocessing_pool)

    # Saves first batch
    # if save_batch_example:
    #  save_batch(config.sample_tf1_batch_location, plain_images_batch)

    #  if config.crop_network:
    #    save_batch(config.sample_tf1_batch_location_cropped, plain_images_batch_cropped)

    curr_batch_sz = plain_images_batch.size(0)

    for dataloader_idx in range(config.num_dataloaders):

        transformed_images_batch = transform_list(current_batch, tf2, preprocessing_pool)

        if config.crop_network:
            transformed_images_batch_cropped = transform_list(current_batch_cropped, tf2, preprocessing_pool)

        # if save_batch_example:
        #  save_batch(config.sample_tf2_batch_location, transformed_images_batch)

        #  if config.crop_network:
        #    save_batch(config.sample_tf2_batch_location_cropped, transformed_images_batch_cropped)
        #  save_batch_example = False

        actual_batch_start = dataloader_idx * curr_batch_sz
        actual_batch_end = actual_batch_start + curr_batch_sz

        plain_images_final_batch[actual_batch_start:actual_batch_end, :, :, :] = plain_images_batch.cuda()
        transformed_images_final_batch[actual_batch_start:actual_batch_end, :, :, :] = transformed_images_batch.cuda()

        if config.crop_network:
            plain_images_final_batch_cropped[actual_batch_start:actual_batch_end, :, :,
            :] = plain_images_batch_cropped.cuda()
            transformed_images_final_batch_cropped[actual_batch_start:actual_batch_end, :, :,
            :] = transformed_images_batch_cropped.cuda()

    if not (curr_batch_sz == config.dataloader_batch_sz):
        print("last batch sz %d" % curr_batch_sz)

    # Cuts excess space
    curr_total_batch_sz = curr_batch_sz * config.num_dataloaders
    plain_images_final_batch = plain_images_final_batch[:curr_total_batch_sz, :, :, :]
    transformed_images_final_batch = transformed_images_final_batch[:curr_total_batch_sz, :, :, :]
    if config.crop_network:
        plain_images_final_batch_cropped = plain_images_final_batch_cropped[:curr_total_batch_sz, :, :, :]
        transformed_images_final_batch_cropped = transformed_images_final_batch_cropped[:curr_total_batch_sz, :, :, :]

    # Makes sobel processing
    plain_images_final_batch = sobel_process(plain_images_final_batch, config.include_rgb)
    transformed_images_final_batch = sobel_process(transformed_images_final_batch, config.include_rgb)
    if config.crop_network:
        plain_images_final_batch_cropped = sobel_process(plain_images_final_batch_cropped, config.include_rgb)
        transformed_images_final_batch_cropped = sobel_process(transformed_images_final_batch_cropped,
                                                               config.include_rgb)

    # Apply cutout after sobel processing to avoid marking its contours
    # Apply it only to transformed images
    if config.cutout:
        transformed_images_final_batch = custom_cutout(transformed_images_final_batch, config.cutout_holes,
                                                       config.cutout_max_box)
        if config.crop_network:
            transformed_images_final_batch_cropped = custom_cutout(transformed_images_final_batch_cropped,
                                                                   config.cutout_holes, config.cutout_max_box)

    # Build result object
    result = {}
    result["plain"] = plain_images_final_batch
    result["transformed"] = transformed_images_final_batch
    result["plain_cropped"] = None
    result["transformed_cropped"] = None
    if config.crop_network:
        result["plain_cropped"] = plain_images_final_batch_cropped
        result["transformed_cropped"] = transformed_images_final_batch_cropped

    return result


'''
def train_epoch_standard(net, optimizer, head, current_epoch, current_k, mse_probability_bounds, train_dataloader, config, tf1, tf2, crop_transform, preprocessing_pool):
    \'''
    Performs a training epoch with loss functions that do not require
    :param net:
    :param optimizer:
    :param head:
    :param current_epoch:
    :param current_k:
    :param mse_probability_bounds:
    :param train_dataloader:
    :param config:
    :param tf1:
    :param tf2:
    :param crop_transform:
    :param preprocessing_pool:
    :return:
    \'''
    for batch_idx, current_batch in enumerate(train_dataloader):
        net.module.zero_grad()

        batches = build_train_batch(current_batch, config, tf1, tf2, crop_transform, preprocessing_pool)

        plain_images_batch = batches["plain"]
        transformed_images_batch = batches["transformed"]
        if config.crop_network:
            plain_images_batch_cropped = batches["plain_cropped"]
            transformed_images_batch_cropped = batches["plain_cropped"]

        # If the network is a cropped network pass also the cropped images versions
        if not config.crop_network:
            plain_network_outputs = net(plain_images_batch, head=head)
            transformed_network_outputs = net(transformed_images_batch, head=head)
        else:
            plain_network_outputs = net(plain_images_batch, plain_images_batch_cropped, head=head)
            transformed_network_outputs = net(transformed_images_batch, transformed_images_batch_cropped,
                                              head=head)

        # Code had support for multihead, but we use only the first
        plain_network_outputs = plain_network_outputs[0]
        transformed_network_outputs = transformed_network_outputs[0]

        label_smoothing_loss = None
        if config.loss == "entropy_trace":
            loss, loss_no_lamb, label_smoothing_loss = IID_entropy_trace_loss(plain_network_outputs,
                                                                              transformed_network_outputs,
                                                                              current_k, config, current_epoch)
        elif config.loss == "vanilla":
            loss, loss_no_lamb = IID_loss(plain_network_outputs, transformed_network_outputs, lamb=config.lamb)
        elif config.loss == "mse":
            loss, loss_no_lamb = IID_entropy_mse_loss(plain_network_outputs, transformed_network_outputs,
                                                      current_k, mse_probability_bounds[0],
                                                      mse_probability_bounds[1], config)
        else:
            raise Exception("Unknown loss type {}".format(config.loss))

        if (batch_idx % (len(train_dataloader) // 20)) == 0:
            print("  [{}/{}] Head {}, Loss {:.3f}, Loss no lamb {:.3f}, {}".format(batch_idx,
                                                                                   len(train_dataloader), head,
                                                                                   loss.item(),
                                                                                   loss_no_lamb.item(),
                                                                                   datetime.now()))
            sys.stdout.flush()

        if not np.isfinite(loss.item()):
            print("Loss is not finite... %s:" % str(loss))
            exit(1)

        loss.backward()
        optimizer.step()

        smoothing_loss = None
        if label_smoothing_loss is not None:
            smoothing_loss = label_smoothing_loss.item()

        return loss.item(), loss_no_lamb.item(), smoothing_loss

def estimate_joint_probability_matrix(net, current_batch, head, config, tf1, tf2, preprocessing_pool):
    \'''
    Gets an estimate of the joint probability matrix based on the provided batch

    :param net: The model
    :param current_batch: Batch of images to use for the estimate in the form (pil image, label)
    :param head: Head for which to get the estimation
    :param config: The training configuration
    :param tf1: plain images transform
    :param tf2: transformed images transform
    :return: torch tensor representing the sum over each sample of all the estimations of the joint probability matrix.
             the resulting matrix must be simmetrized and normalized
    \'''

    # Does not support yet cropped network
    assert(not config.crop_network)
    net.eval()

    # Accumulates the partial sums that will build the final probability matrix
    partial_prob_matrix = None
    current_batch_length = len(current_batch)

    with torch.no_grad():
        # Foreach small batch. Must use dataloader_batch_sz to account for the expansion given by multiple transfromations per image
        for current_batch_index in range(0, current_batch_length, config.dataloader_batch_sz):
            current_small_batch = current_batch[current_batch_index, min(current_batch_index + config.dataloader_batch_sz, current_batch_length)]

            train_batches = build_train_batch(current_small_batch, config, tf1, tf2, None, preprocessing_pool)
            plain_images_batch = train_batches["plain"]
            transformed_images_batch = train_batches["transformed"]

            plain_network_outputs = net(plain_images_batch, head=head)
            transformed_network_outputs = net(transformed_images_batch, head=head)

            _ , out_classes = plain_network_outputs.size()

            # Build joint probability estimates
            joint_probabilities_sum = plain_network_outputs.unsqueeze(2) * transformed_network_outputs.unsqueeze(1)  # config.batch_sz, k, k
            joint_probabilities_sum = joint_probabilities_sum.sum(dim=0)

            if partial_prob_matrix is None:
                partial_prob_matrix = joint_probabilities_sum
            else:
                partial_prob_matrix += joint_probabilities_sum

    return partial_prob_matrix

    net.train()
'''


def update_joint_probability_estimations(plain_network_outputs, transformed_network_outputs,
                                         joint_probabilities_estimations, config):
    '''
    Updates the sequence of estimations by adding the estimation for the current outputs.
    Computataion is detached from the original graph to avoid gradient propagation

    :param plain_network_outputs:
    :param transformed_network_outputs:
    :param joint_probabilities_estimations:
    :param config:
    :return:
    '''

    joint_probabilities_sum = plain_network_outputs.detach().unsqueeze(
        2) * transformed_network_outputs.detach().unsqueeze(1)  # config.batch_sz, k, k
    joint_probabilities_sum = joint_probabilities_sum.sum(dim=0)

    joint_probabilities_estimations.append(joint_probabilities_sum)
    if len(joint_probabilities_estimations) > config.estimation_batches:
        joint_probabilities_estimations.pop(0)


def initialize_batch_norm_layers(config, net, test_dataloader, tf3, crop_transform, preprocessing_pool):
    '''
    Initializes the batch norm layers by making a pass over the test data
    :param config:
    :param net:
    :param test_dataloader:
    :param tf3:
    :param crop_transform:
    :param preprocessing_pool:
    :return:
    '''
    # If the network is freshly initialized, then make a pass over the data in train mode
    # to initialize batch norm layers

    # Enables batch norm statistics recording
    net.train()
    for b_i, batch in enumerate(test_dataloader):

        images, images_cropped = build_eval_batch(batch, config, tf3, crop_transform, preprocessing_pool, True)

        with torch.no_grad():
            # If the network needs cropped images pass also the cropped ones
            if config.crop_network:
                x_outs = net(images, images_cropped)
            else:
                x_outs = net(images)
        # Forces evaluation
        for result in x_outs:
            result.cpu().numpy()


def create_directories(config):
    # Creates initial directories
    config.out_dir = os.path.join(config.out_root, str(config.model_ind))
    if not os.path.exists(config.out_dir):
        os.mkdir(config.out_dir)

    config.tensorboard_directory = os.path.join(config.out_dir, "runs")
    if not os.path.exists(config.tensorboard_directory):
        os.mkdir(config.tensorboard_directory)

    config.checkpoint_directory = os.path.join(config.out_dir, "checkpoints")
    if not os.path.exists(config.checkpoint_directory):
        os.mkdir(config.checkpoint_directory)
    config.samples_directory = os.path.join(config.out_dir, "samples")
    if not os.path.exists(config.samples_directory):
        os.mkdir(config.samples_directory)

    config.cluster_example_directory = os.path.join(config.out_dir, "cluster_examples")
    if not os.path.exists(config.cluster_example_directory):
        os.mkdir(config.cluster_example_directory)
    config.tensorlog_directory = os.path.join(config.out_dir, "tensorlog")
    if not os.path.exists(config.tensorlog_directory):
        os.mkdir(config.tensorlog_directory)
    config.visualizations_directory = os.path.join(config.out_dir, "visualizations")
    if not os.path.exists(config.visualizations_directory):
        os.mkdir(config.visualizations_directory)

    config.sample_tf1_batch_location = os.path.join(config.out_dir, "tf1_batch_sample")
    config.sample_tf2_batch_location = os.path.join(config.out_dir, "tf2_batch_sample")
    config.sample_tf1_batch_location_cropped = config.sample_tf1_batch_location + "_cropped"
    config.sample_tf2_batch_location_cropped = config.sample_tf2_batch_location + "_cropped"
    if not os.path.exists(config.sample_tf1_batch_location):
        os.mkdir(config.sample_tf1_batch_location)
    if not os.path.exists(config.sample_tf2_batch_location):
        os.mkdir(config.sample_tf2_batch_location)
    if not os.path.exists(config.sample_tf1_batch_location_cropped):
        os.mkdir(config.sample_tf1_batch_location_cropped)
    if not os.path.exists(config.sample_tf2_batch_location_cropped):
        os.mkdir(config.sample_tf2_batch_location_cropped)


def check_and_setup_cluster_twohead_config(config):
    '''
    Checks the validity of the configuration file in case of two head clustering and enriches it
    with additional information
    :param config: the configuratoin file
    '''
    # Indicates whether the network also needs cropped versions of the images
    config.crop_network = False
    if config.arch == "CropNetTwoHead":
        config.crop_network = True

    config.twohead = True

    # Use rgb attribute is for supervised training, impede use
    assert(not hasattr(config, "use_rgb"))
    config.use_rgb = False

    if not config.include_rgb:
        config.in_channels = 2
    else:
        config.in_channels = 5

    assert (config.batch_sz % config.num_dataloaders == 0)
    config.dataloader_batch_sz = config.batch_sz // config.num_dataloaders

    assert (config.mode == "IID")
    assert ("TwoHead" in config.arch)

    # Forces only 1 sub head
    config.num_sub_heads = 1
    # Old code needs to have this parameter and have it match head B
    config.output_k = config.output_k_B

    config.train_shuffle = True
    config.test_shuffle = True


def check_and_setup_ensable_configs(global_config, network_configs):
    '''
    Checks the validity of the configuration file in case of two head clustering and enriches it
    with additional information
    :param config: the configuratoin file
    '''

    global_config.twohead = True

    assert (global_config.batch_sz % global_config.num_dataloaders == 0)
    global_config.dataloader_batch_sz = global_config.batch_sz // global_config.num_dataloaders

    assert (global_config.mode == "IID")

    global_config.train_shuffle = True
    global_config.test_shuffle = True

    for current_netowrk_config in network_configs:

        assert (current_netowrk_config.loss in (
        'vanilla', 'vanilla_plus', 'vanilla_plus_alpha', 'vanilla_plus_learnable_alpha', 'entropy_trace',
        'entropy_trace_plus_learnable_alpha', 'mse'))

        # Indicates whether the network also needs cropped versions of the images
        current_netowrk_config.crop_network = False
        if current_netowrk_config.arch == "CropNetTwoHead":
            current_netowrk_config.crop_network = True

            # Use rgb attribute is for supervised training, impede use
            assert (not hasattr(current_netowrk_config, "use_rgb"))
            current_netowrk_config.use_rgb = False

        if not current_netowrk_config.include_rgb:
            current_netowrk_config.in_channels = 2
        else:
            current_netowrk_config.in_channels = 5

        assert ("TwoHead" in current_netowrk_config.arch)

        # Forces only 1 sub head
        current_netowrk_config.num_sub_heads = 1
        # Old code needs to have this parameter and have it match head B
        current_netowrk_config.output_k = current_netowrk_config.output_k_B

    # Copy global settings into single networks
    for key, value in global_config.items():
        if key != "networks":
            for current_netowrk_config in network_configs:
                assert (not key in current_netowrk_config)
                current_netowrk_config[key] = value


def check_and_setup_basic_dial_config(config):
    '''
    Checks the validity of the configuration file in case of two head basic dial clustering and enriches it
    with additional information
    :param config: the configuratoin file
    '''

    if not hasattr(config, "backward_steps"):
        config.backward_steps = 1

    if not hasattr(config, "use_rgb"):
        config.use_rgb = False

    # Privides dot access to dictionary members in transforms
    if hasattr(config, "transforms"):
        new_transforms = []
        for current_transform in config.transforms:
            new_transforms.append(DictWrapper(current_transform))
        config.transforms = new_transforms

    config.in_channels = 2
    if config.use_rgb:
        config.in_channels = 3
    config.domains_count = len(config.domain_datasets)

    if not hasattr(config, "test_datasets"):
        config.test_datasets = []
    if not hasattr(config, "test_datasets_full"):
        config.test_datasets_full = []
    else:
        if not len(config.test_datasets_full) == len(config.test_datasets):
            raise Exception("Length of full test datasets must be the same as the test datasets")
    if not hasattr(config, "map_test_to_domains"):
        config.map_test_to_domains = []

    config.test_domains_count = len(config.test_datasets)
    config.all_domains_count = config.domains_count + config.test_domains_count

    if not hasattr(config, "transformed_is_domain"):
        config.transformed_is_domain = False

    if not hasattr(config, "domain_permutation"):
        # User may ask to use a different mapping between domain id and bn layers
        # Useful for loading models
        config.domain_permutation = list(range(config.all_domains_count))

    if not hasattr(config, "skip_a"):
        config.skip_a = False

    if not hasattr(config, "head_A_epochs"):
        config.head_A_epochs = 1
    if not hasattr(config, "head_B_epochs"):
        config.head_B_epochs = 1
    if not hasattr(config, "cutout"):
        config.cutout = False
    if not hasattr(config, "lamb_annealing_epochs"):
        config.lamb_annealing_epochs = 8

    if not hasattr(config, "domain_specific_lambda"):
        config.domain_specific_lambda = 0.0
    if not hasattr(config, "domain_cluster_lambda_annealing_epochs"):
        config.domain_cluster_lambda_annealing_epochs = 8
    if not hasattr(config, "num_sub_heads"):
        config.num_sub_heads = 1
    if not hasattr(config, "double_eval"):
        config.double_eval = False

    if not hasattr(config, "centroid_alignment_lambda"):
        config.centroid_alignment_lambda = 0.0
    if not hasattr(config, "encoder_features_count"):
        config.encoder_features_count = 512
    if not hasattr(config, "whiten"):
        config.whiten = False
    if not hasattr(config, "centroid_estimation_alpha"):
        config.centroid_estimation_alpha = 0.9
    if not hasattr(config, "centroid_estimation_annealing_epochs"):
        config.centroid_estimation_annealing_epochs = 8
    if not hasattr(config, "domain_specific_bn_affine"):
        config.domain_specific_bn_affine = False
    if not hasattr(config, "domain_cluster_separate_transform"):
        config.domain_cluster_separate_transform = False

    if not hasattr(config, "visualization_epochs"):
        config.visualization_epochs = 10

    assert (config.num_dataloaders == 1) # Do not support multiple dataloaders
    assert (config.batch_sz % config.num_dataloaders == 0)
    config.dataloader_batch_sz = config.batch_sz // config.num_dataloaders

    assert ("TwoHead" in config.encoder_arch)

    # Old code needs to have this parameter and have it match head B
    config.output_k = config.output_k_B

def check_and_setup_generalization_refinement_config(config):
    check_and_setup_basic_dial_config(config)

    if not hasattr(config, "refinement_method"):
        config.refinement_method = "IIC"
    if not hasattr(config, "refinement_entropy_threshold"):
        config.refinement_entropy_threshold = 0.95
    if not hasattr(config, "fixmatch_threshold"):
        config.fixmatch_threshold = 0.95
    # By default no annealing of the threshold. End is set to 1 to avoid division by 0
    if not hasattr(config, "fixmatch_threshold_anneal_start"):
        config.fixmatch_threshold_anneal_start = 0
    if not hasattr(config, "fixmatch_threshold_anneal_end"):
        config.fixmatch_threshold_anneal_end = 1

    if not hasattr(config, "refinement_transforms"):
        raise Exception("In  refinement the 'refinement_transform' must be defined for the test domain")

def perform_evaluation(config, evaluation_manager, statistics_manager, current_epoch, current_step, visualize=False, skip_test_domains=False, prefix=""):

    domains_skipped = False
    with torch.no_grad():

        # Performs evaluation specific to each domain
        for domain_idx in range(config.all_domains_count):

            is_test_domain = domain_idx >= config.domains_count
            # If the current domain is a test domain
            if is_test_domain and skip_test_domains == True:
                print("")
                print("------- Evaluation [{}:{}] Results --------".format(current_epoch, domain_idx))
                print("Double evaluation skipped because current domain is a test domain")
                domains_skipped = True
                continue

            evaluation_results = evaluation_manager.evaluate(domain_idx)

            print("")
            print("------- Evaluation [{}:{}] Results --------".format(current_epoch, domain_idx))
            print("nmi: {}".format(evaluation_results["nmi"]))
            print("mutual_information: {}".format(evaluation_results["mutual_information"]))
            print("conditional_entropy: {}".format(evaluation_results["conditional_entropy"]))
            print("ground_truth_clusters_entropy: {}".format(evaluation_results["ground_truth_clusters_entropy"]))
            print("predicted_clusters_entropy: {}".format(evaluation_results["predicted_clusters_entropy"]))
            if evaluation_results["accuracy"][0] is not None:
                print("accuracy: {}".format(evaluation_results["accuracy"]))

            # Logs statistics
            for key, value in evaluation_results.items():
                if "matrix" not in key:
                    for subhead_index in range(config.num_sub_heads):
                        statistics_manager.update("{}evaluation_domain_{}/{}_subhead_{}".format(prefix, domain_idx, key, subhead_index), value[subhead_index], 0)
            statistics_manager.update("{}evaluation_domain_{}/avg_nmi".format(prefix, domain_idx), mean(evaluation_results["nmi"]), 0)
            if evaluation_results["accuracy"][0] is not None:
                statistics_manager.update("{}evaluation_domain_{}/avg_accuracy".format(prefix, domain_idx), mean(evaluation_results["accuracy"]), 0)

    # Performs evaluation only if all domain information is present
    if not domains_skipped:
        # Performs evaluation computations that need all domain data
        evaluation_results = evaluation_manager.mutual_information_domains_clusters()
        for key, value in evaluation_results.items():
            for subhead_index in range(config.num_sub_heads):
                statistics_manager.update("{}evaluation_all_domains/{}_subhead_{}".format(prefix, key, subhead_index), value[subhead_index], 0)
        if visualize:
            evaluation_manager.compute_latest_features_visualization(os.path.join(config.visualizations_directory, "visualization_step_{}".format(current_step)))