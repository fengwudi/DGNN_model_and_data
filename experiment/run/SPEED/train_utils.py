import warnings
import math
import logging
import pathlib
import subprocess
import time
from hashlib import md5

import numpy as np
import torch
from torch import distributed as dist
import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score


def dist_setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def dist_cleanup():
    dist.destroy_process_group()


def hash_args(**arg_dict):
    paras = [f"{k}={v}" for k, v in sorted(arg_dict.items(), key=lambda x: x[0])]
    paras = ','.join(paras)
    hash = md5(paras.encode('utf-8')).hexdigest()[:6].upper()
    return hash


def check_free_gpus(free_mem_threshold=20000, visible_gpus=None):
    """
    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
    """
    # Get the list of GPUs via nvidia-smi
    smi_query_result = subprocess.check_output("nvidia-smi -q -d Memory | grep -A4 GPU", shell=True)
    # Extract the usage information
    gpu_info = smi_query_result.decode("utf-8").split("\n")
    gpu_info = list(filter(lambda info: "Free" in info, gpu_info))
    gpu_mem_usage = [int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info]
    visible_gpus = {i for i in range(len(gpu_mem_usage))} if visible_gpus is None else set(visible_gpus) 
    free_gpus = sorted([(-mem, i) for i, mem in enumerate(gpu_mem_usage)
                        if (mem >= free_mem_threshold) and (i in visible_gpus)])
    free_gpus = [i for _, i in free_gpus]  # sorted
    return free_gpus


def get_logger(prefix=''):
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    pathlib.Path("log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f'log/{time.strftime("%m%d-%H:%M:%S")}.{prefix}.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class DummyLogger:
    
    def __getattribute__(self, name):
        return lambda *args, **kwargs: None


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)



class EarlyStopMonitor:
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10, *, epoch_start=0):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = epoch_start
        self.best_epoch = epoch_start

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round
