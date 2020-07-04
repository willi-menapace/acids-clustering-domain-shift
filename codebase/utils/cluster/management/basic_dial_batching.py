import torch
import torchvision
from PIL import Image

import statistics

from torchvision.datasets import DatasetFolder

from codebase.utils.cluster.transforms import sobel_make_transforms, sobel_process, custom_cutout, \
    sobel_make_multi_transforms


def list_collate_batcher(batch):
    return [element for element in batch]

def open_image_transform(path):
    return Image.open(path)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class ImagePathFolder(DatasetFolder):
    """
    A Dataset identical to ImageFolder, but that returns paths instead of PIL images
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=lambda path: path, is_valid_file=None):
        super(ImagePathFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples

class BasicDialTrainBatchLoader:
    '''
    Utility class for the creation of training batches
    '''

    def __init__(self, config, processing_pool):
        '''
        Train dataloaders for each domain
        '''

        self.config = config
        self.domain_count = len(config.domain_datasets)
        self.processing_pool = processing_pool

        if hasattr(config, "transforms"):
            tf1, tf2_domain_transforms, _ = sobel_make_multi_transforms(self.config)
        else:
            tf1, tf2, _ = sobel_make_transforms(self.config)
            tf2_domain_transforms = [tf2] * self.domain_count

        # Transforms to apply to images
        self.plain_transform = tf1
        self.augmented_transforms = tf2_domain_transforms


        self.domain_dataloaders = [self.create_dataloader_from_path(domain_path) for domain_path in config.domain_datasets]

        # Iterator dataloaders for each domain
        self.domain_iterators = [iter(domain_dataloader) for domain_dataloader in self.domain_dataloaders]
        self.domain_epochs = [0] * self.domain_count

        multiprocessing_plain_result_map = {domain_id: None for domain_id in range(self.domain_count)}
        multiprocessing_plain_batches = {domain_id: [] for domain_id in range(self.domain_count)}
        multiprocessing_transformed_result_map = {domain_id: None for domain_id in range(self.domain_count)}
        multiprocessing_transformed_batches = {domain_id: [] for domain_id in range(self.domain_count)}

        self.multiprocessing_result_map = {0: multiprocessing_plain_result_map, 1: multiprocessing_transformed_result_map}
        self.multiprocessing_batches = {0: multiprocessing_plain_batches, 1: multiprocessing_transformed_batches}

    def get_avg_steps_per_epoch(self):
        '''
        With multiple domains there is no univoque concept of epoch
        :return: The average number of steps needed to complete an epoch across all domains
        '''

        return int(statistics.mean([len(domain_dataloader) for domain_dataloader in self.domain_dataloaders]))

    def create_dataloader_from_path(self, path, workers=0):
        #dataset = torchvision.datasets.ImageFolder(root=path)
        dataset = ImagePathFolder(root=path)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=int(self.config.dataloader_batch_sz),
                                                   shuffle=True,
                                                   num_workers=workers,
                                                   collate_fn=list_collate_batcher,
                                                   drop_last=False)
        return dataloader

    def get_next_batch_from_domain(self, domain_id):
        '''
        Extracts a batch from the dataloader corresponding to the given id
        :param domain_id: id of the domain for which to fetch batches
        :return: a batch of [(path, gt class)] records
        '''
        while True:
            iterator = self.domain_iterators[domain_id]
            try:
                # Exception will be generated if no other batch is present in the iterator, otherwise return the result
                batch = next(iterator)
                return batch
            except StopIteration:
                # We terminated a dataloader, reset the iterator
                self.domain_epochs[domain_id] += 1
                self.domain_iterators[domain_id] = iter(self.domain_dataloaders[domain_id])

    def tensors_from_batch(self, batch, transform, domain_id, label):
        '''
        Transforms a [(path, gt class)] batch into the form needed for training according to transforms
        :param batch: batch of iamge in the form [(path, gt class), ...]
        :param transforms: transformations to apply to each image
        :param domain_id: id of the domain of the batch images
        :return: image tensor, domain_id tensor, ground_truth tensor
        '''

        images = [row[0] for row in batch]
        ground_truth_tensor = torch.tensor([row[1] for row in batch], dtype=torch.int32).cuda()

        transform = torchvision.transforms.Compose([open_image_transform, transform]) # Add image path opening as first operation

        transformed_list = self.processing_pool.map(transform, images)


        images_tensor = torch.stack(transformed_list).cuda()
        if not self.config.use_rgb:
            images_tensor = sobel_process(images_tensor, include_rgb=False)
        if self.config.cutout:
            images_tensor = custom_cutout(images_tensor, self.config.cutout_holes, self.config.cutout_max_box)

        # Creates the domain tensor
        domain_tensor = torch.zeros((len(batch), 1), dtype=torch.int32)
        domain_tensor.fill_(domain_id)

        return images_tensor, domain_tensor, ground_truth_tensor

    def parallel_tensors_from_batch(self, transform, domain_id, label):
        '''
        Transforms a [(path, gt class)] batch into the form needed for training according to transforms
        :param batch: batch of iamge in the form [(path, gt class), ...]
        :param transforms: transformations to apply to each image
        :param domain_id: id of the domain of the batch images
        :return: image tensor, domain_id tensor, ground_truth tensor
        '''
        raise Exception("Must implement ground truth")
        complete_transform = torchvision.transforms.Compose([open_image_transform, transform])  # Add image path opening as first operation

        # If the computation was already started in the latest cycle, get the result object on which to wait
        result = self.multiprocessing_result_map[label][domain_id]
        if result is not None:
            # Wait for the results to be available
            transformed_list = result.get(None)

        else:
            # No computation was startes, so start one synchronously and get current results
            batch = self.multiprocessing_batches[label][domain_id].pop(0)
            images = [row[0] for row in batch]

            transformed_list = self.processing_pool.map(complete_transform, images)

        # Process current results
        images_tensor = torch.stack(transformed_list).cuda()
        if not self.config.use_rgb:
            images_tensor = sobel_process(images_tensor, include_rgb=False)
        if self.config.cutout:
            images_tensor = custom_cutout(images_tensor, self.config.cutout_holes, self.config.cutout_max_box)

        # Start asynchronous computation for the next batch
        future_batch = self.multiprocessing_batches[label][domain_id].pop(0)
        future_images = [row[0] for row in future_batch]

        future_result = self.processing_pool.map_async(complete_transform, future_images)
        self.multiprocessing_result_map[label][domain_id] = future_result # Memorize object on thich to wait to gather results in the next cycle

        # Creates the domain tensor
        domain_tensor = torch.zeros((len(transformed_list), 1), dtype=torch.int32)
        domain_tensor.fill_(domain_id)

        return images_tensor, domain_tensor

    def get_next_batch(self):

        plain_batches = []
        transformed_batches = []

        # Extract and transfrom batch from each domain
        for domain_id in range(self.domain_count):

            # The parallel computation routines expect al least two batches to be available at the moment of call
            #while len(self.multiprocessing_batches[0][domain_id]) != 2:
            #    batch = self.get_next_batch_from_domain(domain_id)
            #    self.multiprocessing_batches[0][domain_id].append(batch)
            #    self.multiprocessing_batches[1][domain_id].append(batch)
            current_batch = self.get_next_batch_from_domain(domain_id)
            plain_batches.append(self.tensors_from_batch(current_batch, self.plain_transform, domain_id, label=0))
            transformed_batches.append(self.tensors_from_batch(current_batch, self.augmented_transforms[domain_id], domain_id, label=1))

            #plain_batches.append(self.parallel_tensors_from_batch(self.plain_transform, domain_id, label=0))
            #transformed_batches.append(self.parallel_tensors_from_batch(self.augmented_transform, domain_id, label=1))

        return BasicDialTrainBatch(plain_batches, transformed_batches)

class BasicDialTrainBatch:
    '''
    Training batch
    '''

    def __init__(self, plain_batches, transformed_batches):
        '''
        Creates a train batch with the provided plain and trasnformed batches for each domain
        :param plain_batches:  list with a (plain_images_tensor, domain_tensor, ground_truth tensor) element for each domain
        :param transformed_batches: list with a (transformed_images_tensor, domain_tensor, ground_truth tensor) element for each domain
        '''
        self.plain_batches = plain_batches
        self.transformed_batches = transformed_batches

    def get_batches_by_domain(self, domain_id):
        '''
        Gets the training batches for the given domain
        :param domain_id:
        :return: (plain_images_tensor, domain_tensor), (transformed_images_tensor, domain_tensor)
        '''
        return self.plain_batches[domain_id], self.transformed_batches[domain_id]





