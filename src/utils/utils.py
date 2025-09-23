import random
import subprocess
from contextlib import ContextDecorator
from typing import Callable
import time

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class timeit(ContextDecorator):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        elapsed = time.time() - self.start
        print(f'{self.name} took {elapsed:.2f} seconds')
        return False  # re-raise any exception that occurred in the with block


# kill all wandb processes – sometimes required due a bug in wandb
def kill_all_wandb_processes():
    subprocess.run('ps aux|grep wandb|grep -v grep | awk \'{print $2}\'|xargs kill -9', shell=True)
