import numpy as np

class TensorRecord:

    def __init__(self, name):
        self.name = name
        # Retains all tensors without ground truth data
        self.tensors_list = []
        # Retains ground_truth_class -> list of tensors mappings
        # for tensors with associated ground truth
        self.ground_truth_tensors_map = {}

        self.tensor_size = None

    def _add_tensor_to_list(self, tensor, list):
        '''
        Adds each batch sample in tensor to list
        :param tensor:
        :param list:
        :return:
        '''
        for batch_sample_idx in range(tensor.shape[0]):
            list.append(tensor[batch_sample_idx])

    def _same_size(self, t1, t2):
        '''
        Returns true if all the dimensions apart from the batch dimenion are the same in the size arguments
        :param t1: size of the first tensor
        :param t2: size of the second tensor
        :return:
        '''
        for i in range(1, len(t1)):
            if t1[i] != t2[i]:
                return False
        return True

    def add_tensor(self, tensor, ground_truth_tensor=None):
        '''
        Registers tensor information along with its optional ground truth information. The shape of the registered
        tensors may vary only in batch size during successive calls.
        :param tensor: the tensor with shape (batch_size, ....) to register
        :param ground_truth_tensor: optional tensor with shape (batch_size) containing ground truth classes
        :return:
        '''

        if ground_truth_tensor is not None:
            if tensor.size()[0] != ground_truth_tensor.size()[0]:
                raise Exception("Data tensor has batch size {} but ground truth tensor has batch size {}".format(tensor.size()[0], ground_truth_tensor.size()[0]))

        if self.tensor_size is None:
            self.tensor_size = tensor.size()
        else:
            current_tensor_size = tensor.size()
            if not self._same_size(self.tensor_size, current_tensor_size):
                raise Exception("Originally stored tensors of size {}, but received a tensor of size {}".format(self.tensor_size, tensor.size()))

        # Transfer tensor to numpy
        tensor = tensor.cpu().numpy()

        if ground_truth_tensor is None:
            self._add_tensor_to_list(tensor, self.tensors_list)
        else:
            ground_truth_tensor = ground_truth_tensor.cpu().numpy()

            # If ground truth data is provided register each tensor in the list corresponding to its ground truth class
            for batch_sample_idx in range(tensor.shape[0]):
                current_tensor = tensor[batch_sample_idx]
                current_ground_truth_class = ground_truth_tensor[batch_sample_idx]

                if not current_ground_truth_class in self.ground_truth_tensors_map:
                    self.ground_truth_tensors_map[current_ground_truth_class] = []
                self.ground_truth_tensors_map[current_ground_truth_class].append(current_tensor)

    def get_tensors_by_class(self, ground_truth_class):
        tensors = self.ground_truth_tensors_map[ground_truth_class]
        tensors = np.array(tensors)
        return tensors

    def get_ground_truth_classes(self):
        '''
        :return: sequence of ground truth classes in ascending order
        '''
        return list(sorted(self.ground_truth_tensors_map.keys()))

    def get_tensors(self):
        all_tensors = self.tensors_list

        # Extracts all labeled tensors and adds them to the tensor list
        for current_tensors_list in self.ground_truth_tensors_map.values():
            all_tensors.extend(current_tensors_list)

        return np.array(all_tensors)

class CompressedTensorRecord(TensorRecord):

    def __init__(self):
        super(TensorRecord, self).__init__()

    def add_tensor(self, tensor, ground_truth_tensor=None):

        # If the tensor comes from a 2D convolution, then store only pooled information
        if len(self.tensor_size) == 4:
            tensor = np.mean(tensor, axis=3)
            tensor = np.mean(tensor, axis=2)

        super(TensorRecord, self).add_tensor(tensor, ground_truth_tensor)