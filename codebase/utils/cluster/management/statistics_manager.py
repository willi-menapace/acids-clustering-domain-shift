import numpy as np

from tensorboardX import SummaryWriter

class StatisticsManager:
    '''
    Manages training statistics and provides an interface to tensorboard
    '''

    def __init__(self, config, default_smoothing_alpha=0.93, writer=None):
        '''
        Initializes the manager opening a tensorboard session
        :param config: the configuration file
        :param default_smoothing_alpha: alpha factor to use for the exponential running mean of each logged value
        :param preconfigured writer
        '''
        # Initializes tensorbaord writer
        if writer is None:
            self.writer = SummaryWriter(config.tensorboard_directory, comment="-")
        else:
            self.writer = writer

        self.default_smoothing_alpha = default_smoothing_alpha
        self.registered_values = {}

    def update(self, key, value, smoothing_alpha=None):
        '''
        Updates the value associated to the specified key. If key is not registered it is automatically added

        :param key: key for which to store the value
        :param value: value to store
        :param smoothing_alpha: alpha factor for exponential average. If None the default smoothing factor is applied
        :return:
        '''

        if smoothing_alpha is None:
            smoothing_alpha = self.default_smoothing_alpha

        if not key in self.registered_values:
            self.registered_values[key] = value
        else:
            old_value = self.registered_values[key]
            self.registered_values[key] = (old_value * smoothing_alpha) + (value * (1 - smoothing_alpha))

    def get_value(self, key):
        '''
        Retrieves the value associated to the given key
        :param key:
        :return:
        '''
        if not key in self.registered_values:
            raise Exception("Requested value for key '{}', but no value is registered under that key".format(key))

        return self.registered_values[key]

    def write_to_tensorboard(self, step):
        '''
        Writes all the stored values to the tensorboard log
        :param step:
        :return:
        '''
        for key, value in self.registered_values.items():
            if type(value) is np.ndarray:
                self.writer.add_histogram(key, value, step)
            else:
                self.writer.add_scalar(key, value, step)