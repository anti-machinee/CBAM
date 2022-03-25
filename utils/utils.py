import os
from pathlib import Path
import argparse
import json
import yaml
from collections import OrderedDict
from dotmap import DotMap
from itertools import repeat

import pandas as pd


def get_args():
    args = argparse.ArgumentParser(__doc__)
    args.add_argument("-c", "--config", default="", type=str,
                      help="config file path")
    args.add_argument("--resume-model", default=None,
                      help="path to latest model checkpoint")
    args.add_argument("--resume-loss", default=None, type=str,
                      help="path to latest loss checkpoint")
    args.add_argument("--local_rank", default=0, type=int,
                      help="node rank for distributed training")
    args.add_argument("--freq-log", default=100, type=int,
                      help="number of times to log while training")
    args = args.parse_args()
    return args


def get_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError("Could not find config file {}".format(config_path))
    with open(config_path, "r") as config_file:
        if config_path.endswith(".yaml"):
            config_dict = yaml.safe_load(config_file)
        else:
            config_dict = json.load(config_file, object_hook=OrderedDict)
    config = DotMap(config_dict)
    return config, config_dict


def save_config(config_dict, save_path):
    with open(save_path, "w") as f:
        if save_path.endswith(".yaml"):
            yaml.dump(config_dict, f, indent=4, sort_keys=False)
        else:
            json.dump(config_dict, f, indent=4, sort_keys=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def make_dir(path, is_delete=False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if is_delete:
            os.system(f"rm -r {path}/*")


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
