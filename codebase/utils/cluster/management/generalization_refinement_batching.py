import statistics

import torch
import torchvision

from codebase.utils.cluster.management.basic_dial_batching import ImagePathFolder, list_collate_batcher, \
    open_image_transform
from codebase.utils.cluster.transforms import sobel_make_refinement_transforms, sobel_process, custom_cutout


class RefinementTrainBatchLoader:
    '''
    Utility class for the creation of training batches
    '''

    def __init__(self, config, processing_pool):
        '''
        Train dataloaders for each domain
        '''

        self.config = config
        self.domain_count = len(config.domain_datasets)
        self.test_domain_count = config.test_domains_count
        self.processing_pool = processing_pool


        tf1, tf2_domain_transforms, _ = sobel_make_refinement_transforms(self.config)


        # Transforms to apply to images
        self.plain_transform = tf1
        self.augmented_transforms = tf2_domain_transforms


        self.domain_dataloaders = [self.create_dataloader_from_path(domain_path) for domain_path in config.test_datasets]

        # Iterator dataloaders for each domain
        self.domain_iterators = [iter(domain_dataloader) for domain_dataloader in self.domain_dataloaders]
        self.domain_epochs = [0] * self.test_domain_count

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

        # Test datasets use a range of domain ids after the train domains
        target_domain_id = self.domain_count + domain_id
        domain_tensor.fill_(target_domain_id)

        return images_tensor, domain_tensor, ground_truth_tensor

    def get_next_batch(self):

        plain_batch_images = []
        plain_batch_domains = []
        plain_batch_ground_truth = []
        transformed_batch_images = []
        transformed_batch_domains = []
        transformed_batch_ground_truth = []

        # Extract and transfrom batch from each domain
        for domain_id in range(self.test_domain_count):

            current_batch = self.get_next_batch_from_domain(domain_id)
            current_image, current_domain, current_ground_truth = self.tensors_from_batch(current_batch, self.plain_transform, domain_id, label=0)
            plain_batch_images.append(current_image)
            plain_batch_domains.append(current_domain)
            plain_batch_ground_truth.append(current_ground_truth)

            current_image, current_domain, current_ground_truth = self.tensors_from_batch(current_batch, self.augmented_transforms[domain_id], domain_id, label=1)
            transformed_batch_images.append(current_image)
            transformed_batch_domains.append(current_domain)
            transformed_batch_ground_truth.append(current_ground_truth)

        plain_batch = (torch.cat(plain_batch_images, dim=0), torch.cat(plain_batch_domains, dim=0), torch.cat(plain_batch_ground_truth, dim=0))
        transformed_batch = (torch.cat(transformed_batch_images, dim=0), torch.cat(transformed_batch_domains, dim=0), torch.cat(transformed_batch_ground_truth, dim=0))

        return RefinementTrainBatch(plain_batch, transformed_batch)

class RefinementTrainBatch:
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

    def get_batches(self):
        '''
        Gets the training batches
        :return: tuples (plain_images_tensor, domain_tensor), (transformed_images_tensor, domain_tensor) representing
                 the training batches of plain and augmented images. Images from the same domain are contiguous
        '''
        return self.plain_batches, self.transformed_batches