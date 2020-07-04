import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

class TensorPlotter:

    def __init__(self):
        pass

    @staticmethod
    def _compute_density(values, points=200):
        '''
        Computes the density of the input values
        :param values: flat numerical sequence of which to compute the density
        :param points: the number of density points to produce
        :return: x, y sequences of length points representing the density plot
        '''

        # Computes function density
        density = gaussian_kde(values)
        density._compute_covariance()

        xs = np.linspace(np.min(values), np.max(values), points)
        density_xs = density(xs)
        return xs, density_xs

    @staticmethod
    def plot_linear_density(idx, tensor_record, filename=None):
        '''
        Plots the density of neuron idx in the tensor
        :param idx: id of the neuron to plot
        :param tensor_record: tensor values of shape (1, neurons)
        :param filename: filename representing plot save location, if None the plot is displayed
        :return:
        '''

        tensor = tensor_record.get_tensors()
        # Slices the data relative to the indexed neuron
        tensor = tensor[:, idx]
        tensor = np.reshape(tensor, -1)

        xs, density_xs = TensorPlotter._compute_density(tensor)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xs, density_xs)
        ax.set_ylim(0, 1)
        if filename:
            fig.savefig(filename)
            plt.close(fig)
        else:
            plt.show()


    @staticmethod
    def plot_linear_density_by_ground_truth(idx, tensor_record, filename=None):
        '''
        Plots the density of neuron idx in the tensor
        :param idx: id of the neuron to plot
        :param tensor_record: tensor values of shape (1, neurons)
        :param filename: filename representing plot save location, if None the plot is displayed
        :return:
        '''

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ground_truth_labels = tensor_record.get_ground_truth_classes()
        for current_class in ground_truth_labels:

            tensor = tensor_record.get_tensors_by_class(current_class)
            # Slices the data relative to the indexed neuron
            tensor = tensor[:, idx]
            tensor = np.reshape(tensor, -1)
            try:
                #Avoids computing density over a too small set
                if tensor.size > 20:
                    # Computes function density
                    xs, density_xs = TensorPlotter._compute_density(tensor)

                    ax.plot(xs, density_xs, label=str(current_class))
                    ax.set_ylim(0, 1)
                    ax.legend()

                if filename:
                    fig.savefig(filename)
                    plt.close(fig)
                else:
                    plt.show()
            except:
                print("Warning: density calculation for tensor {}:{} failed".format(tensor_record.name, idx))

    @staticmethod
    def plot_channel_density(idx, tensor_record, filename=None):
        '''
        Plots the density of the average value of channel idx
        :param idx: id of the channel to plot
        :param tensor_record: tensor record containing values of shape (1, channels, rows, columns)
        :param filename: filename representing plot save location, if None the plot is displayed
        :return:
        '''

        tensor = tensor_record.get_tensors()
        # Slices the data relative to the indexed channel
        tensor = tensor[:, idx]
        # Reduces channel to a single value per sample
        tensor = np.mean(tensor, axis=2)
        tensor = np.mean(tensor, axis=3)
        tensor = np.reshape(tensor, -1)

        xs, density_xs = TensorPlotter._compute_density(tensor)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xs, density_xs)
        ax.set_ylim(0, 1)
        if filename:
            fig.savefig(filename)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_channel_density_by_ground_truth(idx, tensor_record, filename=None):
        '''
        Plots the density of neuron idx in the tensor
        :param idx:
        :param tensor_record: tensor values of shape (1, channels, rows, columns)
        :param filename: filename representing plot save location, if None the plot is displayed
        '''

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ground_truth_labels = tensor_record.get_ground_truth_classes()
        for current_class in ground_truth_labels:

            tensor = tensor_record.get_tensors_by_class(current_class)
            # Slices the data relative to the indexed neuron
            tensor = tensor[:, idx]
            # Reduces channel to a single value per sample
            tensor = np.mean(tensor, axis=2)
            tensor = np.mean(tensor, axis=1)
            tensor = np.reshape(tensor, -1)

            if tensor.size > 1:
                # Computes function density
                xs, density_xs = TensorPlotter._compute_density(tensor)

                ax.plot(xs, density_xs, label=str(current_class))
                ax.set_ylim(0, 1)
                ax.legend()

            if filename:
                fig.savefig(filename)
                plt.close(fig)
            else:
                plt.show()
