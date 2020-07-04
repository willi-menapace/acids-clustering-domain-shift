import torch

class BasicDialStateManager:
    '''
    Manages persistence of the training process
    '''

    def __init__(self):
        pass

    def load(self, checkpoint_path, network_manager, optimizer_manager, discard_optimizer=False):
        '''
        Loads checkpointed state
        :param checkpoint_path: path to load the checkpoint from
        :param network_manager: network manager to load parameters into
        :param optimizer_manager: optimizer manager to load parameters into
        :return training step
        '''
        # Load all states
        state_dict = torch.load(checkpoint_path)
        network_manager.load(state_dict)
        if discard_optimizer == False:
            optimizer_manager.load(state_dict)
        step = state_dict["step"]

        return step

    def save(self, checkpoint_path, step, network_manager, optimizer_manager):
        '''
        Saves state to a checkpoint
        :param checkpoint_path: path to which to save the checkpoint
        :param step: current training step
        :param network_manager:
        :param optimizer_manager:
        :return:
        '''

        # Build state dictionary
        state_dict = {}
        state_dict["step"] = step
        state_dict.update(network_manager.get_state())
        state_dict.update(optimizer_manager.get_state())

        torch.save(state_dict, checkpoint_path)
