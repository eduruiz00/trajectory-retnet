import time
import sys
import os
import glob
import pickle
import json
import torch
import pdb


def mkdir(savepath, prune_fname=False):
    """
    returns `True` iff `savepath` is created
    """
    if prune_fname:
        savepath = os.path.dirname(savepath)
    if not os.path.exists(savepath):
        try:
            os.makedirs(savepath)
        except:
            print(f'[ utils/serialization ] Warning: did not make directory: {savepath}')
            return False
        return True
    else:
        return False


def get_latest_epoch(loadpath):
    """
    Get the latest epoch from a directory of saved states.
    """
    states = glob.glob1(loadpath, 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch


def load_model(*loadpath, epoch=None, device='cuda:0'):
    """
    Load a model from a directory of saved states.
    """
    loadpath_all = os.path.join(*loadpath)
    config_path = os.path.join(loadpath_all, 'model_config.pkl')
    if not os.path.isfile(config_path):
        loadpath_all = os.path.join(*loadpath)
        loadpath_all = os.path.join(loadpath_all, os.listdir(loadpath_all)[-1])
        config_path = os.path.join(loadpath_all, 'model_config.pkl')

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath_all)

    print(f'[ utils/serialization ] Loading model epoch: {epoch}')
    state_path = os.path.join(loadpath_all, f'state_{epoch}.pt')

    config = pickle.load(open(config_path, 'rb'))
    state = torch.load(state_path)

    model = config()
    model.to(device)
    model.load_state_dict(state, strict=True)

    print(f'\n[ utils/serialization ] Loaded config from {config_path}\n')
    print(config)

    return model, epoch


def load_config(*loadpath):
    """
    Load a config from a pickle file.
    """
    loadpath_all = os.path.join(*loadpath)
    if not os.path.isfile(loadpath_all):
        loadpath_all = os.path.join(*loadpath[:-1])
        loadpath_all = os.path.join(loadpath_all, os.listdir(loadpath_all)[-1])
        loadpath_all = os.path.join(loadpath_all, loadpath[-1])
    config = pickle.load(open(loadpath_all, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath_all}')
    print(config)
    return config


def load_from_config(*loadpath):
    """
    Load a config and make an instance of the class.
    """
    config = load_config(*loadpath)
    return config.make()


def load_args(*loadpath):
    """
    Load a config and make an instance of the class.
    """
    from .setup import Parser
    loadpath = os.path.join(*loadpath)
    args_path = os.path.join(loadpath, 'args.json')
    args = Parser()
    args.load(args_path)
    return args
