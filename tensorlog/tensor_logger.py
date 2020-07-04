from tensorlog.tensor_record import TensorRecord


class TensorLogger:

    def __init__(self):
        # Map from name to tensor loggers
        self.tensor_loggers_map = {}
        self.enabled = True

    def clear(self):
        # Deletes all records memorized by the logger
        self.tensor_loggers_map = {}

    def get_tensor_record_by_name(self, name):
        return self.tensor_loggers_map.get(name)

    def add_tensor(self, name, tensor, ground_truth_tensor=None):
        '''
        Registers tensor information along with its optional ground truth information
        :param name: the name of the tensor record where to add the tensor
        :param tensor: the tensor with shape (batch_size, ....) to register
        :param ground_truth_tensor: optional tensor with shape (batch_size) containing ground truth classes
        :return:

        '''

        if not self.enabled:
            return

        if name not in self.tensor_loggers_map:
            self.tensor_loggers_map[name] = TensorRecord(name)
        tensor_record = self.tensor_loggers_map[name]
        tensor_record.add_tensor(tensor, ground_truth_tensor)
