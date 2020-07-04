import torch


def save_checkpoint(net, optimizer, epoch, estimators, checkpoint_name):
    save_dict = {"model": net.module.state_dict(),
         "optimizer": optimizer.state_dict(),
         # The epoch would be immediately incremented after saving, so save next epoch
         "epoch": epoch + 1}

    if estimators:
        save_dict["estimator_A"] = estimators["A"].state_dict()
        save_dict["estimator_B"] = estimators["B"].state_dict()

    torch.save(save_dict, checkpoint_name)


def save_ensemble_checkpoint(all_networks, optimizer, epoch, all_network_estimators, checkpoint_name):
    save_dict = dict()

    save_dict["optimizer"] = optimizer.state_dict()
    # The epoch would be immediately incremented after saving, so save next epoch
    save_dict["epoch"] = epoch + 1

    for idx, current_network in enumerate(all_networks):
        save_dict["model_{}".format(idx)] = current_network.module.state_dict()

        # Saves estimators if the network has them
        if idx in all_network_estimators:
            save_dict["estimator_{}_A".format(idx)] = all_network_estimators[idx]['A'].state_dict()
            save_dict["estimator_{}_B".format(idx)] = all_network_estimators[idx]['B'].state_dict()

    torch.save(save_dict, checkpoint_name)


def load_checkpoint(checkpoint_name):
    checkpoint = torch.load(checkpoint_name)

    return checkpoint
