import sys
import os

import torch
import torchvision
from torch.utils.data import ConcatDataset

from PIL import Image
import numpy as np

from codebase.utils.cluster.transforms import sobel_make_transforms


# Used by sobel and greyscale clustering twohead scripts -----------------------

def cluster_simple_dataloader(config):
    '''
    Returns a test dataloader. Loads all images from a folder and returns them as touples <PIL Image, folderclass>
    '''
    assert (config.mode == "IID")

    assert (config.dataset == "custom")

    dataset_class = torchvision.datasets.ImageFolder

    # datasets produce either 2 or 5 channel images based on config.include_rgb
    tf1, tf2, tf3 = sobel_make_transforms(config)

    train_imgs = torchvision.datasets.ImageFolder(
        root=config.dataset_root,
        transform=None,
        target_transform=None)

    eval_dataloader = torch.utils.data.DataLoader(train_imgs,
                                                  batch_size=int(config.dataloader_batch_sz),
                                                  shuffle=False,
                                                  num_workers=4,
                                                  collate_fn=list_collate_batcher,
                                                  drop_last=False)
    return eval_dataloader, tf3

def cluster_ensemble_create_dataloaders(global_config):
    '''
    Creates train and test dataloaders
    :param global_config: the global configuration
    :return:
    '''
    assert (global_config.mode == "IID")
    assert (global_config.twohead)
    assert (global_config.dataset == "custom")

    head_a_dataloader = _create_dataloaders(global_config,
                                            shuffle=global_config.train_shuffle)

    head_b_dataloader = _create_dataloaders(global_config,
                                            shuffle=global_config.train_shuffle)

    test_dataloader = _create_mapping_loader(global_config,
                                             shuffle=global_config.test_shuffle)

    additional_dataloaders = [_create_mappling_loader_from_path(path, global_config, shuffle=global_config.test_shuffle) for path in
                              global_config.additional_test_datasets]

    all_test_dataloaders = [test_dataloader] + additional_dataloaders

    return {"A": head_a_dataloader, "B": head_b_dataloader}, all_test_dataloaders

def cluster_ensemble_create_transforms(networks_configs):
    '''
    Creates the IIC training and testing transformations for each network
    :param network_config:
    :return: list of transformations for each network
    '''
    all_transforms = []

    for network_config in networks_configs:
        crop_transform_list = [torchvision.transforms.Resize(network_config.standard_image_size)]
        # Starts cropping at the center and then performs a random crop
        crop_transform_list.append(torchvision.transforms.CenterCrop(network_config.standard_image_size[0] / 2))
        crop_transform_list.append(torchvision.transforms.RandomCrop(network_config.cropnet_crop_size))
        crop_transform = torchvision.transforms.Compose(crop_transform_list)

        # datasets produce either 2 or 5 channel images based on config.include_rgb
        tf1, tf2, tf3 = sobel_make_transforms(network_config)

        all_transforms.append((tf1, tf2, tf3, crop_transform))

    return all_transforms

def cluster_twohead_create_dataloaders(config):
    assert (config.mode == "IID")
    assert (config.twohead)
    assert (config.dataset == "custom")

    crop_transform_list = [torchvision.transforms.Resize(config.standard_image_size)]
    # Starts cropping at the center and then performs a random crop
    crop_transform_list.append(torchvision.transforms.CenterCrop(config.standard_image_size[0] / 2))
    crop_transform_list.append(torchvision.transforms.RandomCrop(config.cropnet_crop_size))
    crop_transform = torchvision.transforms.Compose(crop_transform_list)

    # datasets produce either 2 or 5 channel images based on config.include_rgb
    tf1, tf2, tf3 = sobel_make_transforms(config)

    head_a_dataloader = _create_dataloaders(config, shuffle=config.train_shuffle)

    head_b_dataloader = _create_dataloaders(config, shuffle=config.train_shuffle)

    test_dataloader = _create_mapping_loader(config,
                                             shuffle=config.test_shuffle)

    additional_dataloaders = [_create_mappling_loader_from_path(path, config, shuffle=config.test_shuffle) for path in
                              config.additional_test_datasets]

    all_test_dataloaders = [test_dataloader] + additional_dataloaders

    return {"A": head_a_dataloader, "B": head_b_dataloader}, all_test_dataloaders, tf1, tf2, tf3, crop_transform

# Data creation helpers --------------------------------------------------------

# Returns an list of element as a batch for the DataLoader
def list_collate_batcher(batch):
    return [element for element in batch]


def _create_dataloaders(config, shuffle=False):
    assert ("custom" == config.dataset)

    collate_batch = list_collate_batcher

    train_imgs = torchvision.datasets.ImageFolder(root=config.dataset_root)
    train_dataloader = torch.utils.data.DataLoader(train_imgs,
                                                   batch_size=int(config.dataloader_batch_sz),
                                                   shuffle=shuffle,
                                                   num_workers=4,
                                                   collate_fn=collate_batch,
                                                   drop_last=False)

    num_train_batches = len(train_dataloader)
    print("Number of batches per epoch: %d" % num_train_batches)
    sys.stdout.flush()

    return train_dataloader


def _create_mapping_loader(config, shuffle=False):
    return _create_mappling_loader_from_path(config.dataset_root, config, shuffle)


def _create_mappling_loader_from_path(path, config, shuffle=False):
    assert ("custom" == config.dataset)

    dataset = torchvision.datasets.ImageFolder(
        root=path)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=config.batch_sz,
                                             # full batch
                                             shuffle=shuffle,
                                             collate_fn=list_collate_batcher,
                                             num_workers=4,
                                             drop_last=False)

    return dataloader


# Applies a transformation to a list of elements and transforms it in a tensor
def transform_list(current_list, transform, pool):
    # Transforms only the data point and not the label

    # If the list contains the labels then remove them
    if isinstance(current_list[0], tuple):
        current_list = [element[0] for element in current_list]
    transformed_list = pool.map(transform, current_list)

    # If the transformation contains tensors, then stack them together instead of returning a list
    if not isinstance(transformed_list[0], Image.Image):
        return torch.stack(transformed_list)
    else:
        return transformed_list


def save_batch(output_dir, batch):
    for idx, image in enumerate(batch):
        np_image = image.data.numpy()
        # Handles the case where there are also the rgb channels in the image
        if np_image.shape[0] == 4:
            pil_image = Image.fromarray(np.rollaxis((255 * np_image[0:3]).astype(np.uint8), 0, 3))
        else:
            minimum = np_image.min()
            maximum = np_image.max()
            np_image = ((np_image - minimum) / (maximum - minimum)) * 255.0
            pil_image = Image.fromarray(np.rollaxis(np.concatenate([np_image] * 3).astype(np.uint8), 0, 3))
        pil_image.save(os.path.join(output_dir, "{}.png".format(idx)))
