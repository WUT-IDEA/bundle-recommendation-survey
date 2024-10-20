import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.modules.loss
import matplotlib.pyplot as plt
from math import ceil


def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join(
        ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])


def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
            [
                d
                for d in os.listdir(models_dir)
                if os.path.isdir(os.path.join(models_dir, d))
            ]
        ).astype(np.int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                        f"--{param}",
                        action="append",
                        type=type(default[0]),
                        default=default,
                        help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser

def show(metrics_log):
    x = range(0, len(list(metrics_log.values())[0]))
    i = 1
    columns = 2
    rows = ceil(len(metrics_log)/columns)
    for k, v in metrics_log.items():
        plt.subplot(rows, columns, i)
        plt.plot(x, v, '.-')
        plt.title('{} vs epochs'.format(k))
        i += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()


def get_perf(metrics_log, window_size, target, show=True):
    # max
    maxs = {title: 0 for title in metrics_log.keys()}
    assert target in maxs
    length = len(metrics_log[target])
    for v in metrics_log.values():
        assert length == len(v)
    if window_size >= length:
        for k, v in metrics_log.items():
            maxs[k] = np.mean(v)
    else:
        for i in range(length-window_size):
            now = np.mean(metrics_log[target][i:i+window_size])
            if now > maxs[target]:
                for k, v in metrics_log.items():
                    maxs[k] = np.mean(v[i:i+window_size])
    if show:
        for k, v in maxs.items():
            print('{}:{:.5f}'.format(k, v), end=' ')
    return maxs


def check_overfitting(metrics_log, target, threshold=0.02, show=False):
    maxs = get_perf(metrics_log, 1, target, False)
    assert target in maxs
    overfit = (maxs[target]-metrics_log[target][-1]) > threshold
    if overfit and show:
        print('***********overfit*************')
        print('best:', end=' ')
        for k, v in maxs.items():
            print('{}:{:.5f}'.format(k, v), end=' ')
        print('')
        print('now:', end=' ')
        for k, v in metrics_log.items():
            print('{}:{:.5f}'.format(k, v[-1]), end=' ')
        print('')
        print('***********overfit*************')
    return overfit


def early_stop(metric_log, early, threshold=0.01):
    if len(metric_log) >= 2 and metric_log[-1] < metric_log[-2] and metric_log[-1] > threshold:
        return early-1
    else:
        return early

