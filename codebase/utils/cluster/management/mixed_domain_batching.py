import torch

from codebase.utils.cluster.management.basic_dial_batching import ImagePathFolder, BasicDialTrainBatchLoader, \
    list_collate_batcher


class MixedDomainTrainBatchLoader(BasicDialTrainBatchLoader):
    '''
    Batch loader that produces batches with images sampled from all domains
    '''

    def __init__(self, config, processing_pool):
        super(MixedDomainTrainBatchLoader, self).__init__(config, processing_pool)

        if not(self.config.dataloader_batch_sz % self.domain_count == 0):
            raise Exception("The batch size is not divisible by the number of domains")

        self.domain_batch_size = int(self.config.dataloader_batch_sz // self.domain_count)

    def create_dataloader_from_path(self, path, workers=0):


        dataset = ImagePathFolder(root=path)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=int(self.config.dataloader_batch_sz // self.domain_count),
                                                   shuffle=True,
                                                   num_workers=workers,
                                                   collate_fn=list_collate_batcher,
                                                   drop_last=False)
        return dataloader

    def get_next_batch(self):
        '''
        Creates a batch of plain and transformed images taking images sampling from every domain
        :return:  MixedDomainTrainBatch representing the training batch
        '''

        plain_batch_images = []
        plain_batch_domains = []
        plain_batch_ground_truth = []
        transformed_batch_images = []
        transformed_batch_domains = []
        transformed_batch_ground_truth = []

        # Extract and transfrom batch from each domain
        for domain_id in range(self.domain_count):

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

        return MixedDomainTrainBatch(plain_batch, transformed_batch)

class MixedDomainTrainBatch:
    '''
    Training batch containing images sampled from every domain
    '''

    def __init__(self, plain_batch, transformed_batch):
        '''
        Creates a train batch with the provided plain and trasnformed images batch
        :param plain_batches:  tuple (plain_images_tensor, domain_tensor, ground_truth tensor) where images belonging to the same domain are contiguous
        :param transformed_batches: tuple (transformed_images_tensor, domain_tensor, ground_truth tensor) where images belonging to the same domain are contiguous
        '''
        self.plain_batch = plain_batch
        self.transformed_batch = transformed_batch

    def get_batches(self):
        '''
        Gets the training batches
        :param domain_id:
        :return: tuples (plain_images_tensor, domain_tensor), (transformed_images_tensor, domain_tensor) representing
                 the training batches of plain and augmented images. Images from the same domain are contiguous
        '''
        return self.plain_batch, self.transformed_batch