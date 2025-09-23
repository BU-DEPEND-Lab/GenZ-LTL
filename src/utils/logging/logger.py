import argparse
from abc import ABC, abstractmethod

import numpy as np


class Logger(ABC):
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.keys = None

    @abstractmethod
    def log_config(self):
        """
        Logs the given config.
        """
        pass

    @abstractmethod
    def log(self, data: dict[str, float | list[float]]):
        """
        Logs the given data.
        """
        pass

    def finish(self):
        """
        Finishes logging for the current experiment.
        """
        pass

    @staticmethod
    def aggregate(data: dict[str, float | list[float]]) -> dict[str, float]:
        """
        Aggregates values for keys if there are multiple values for a key.
        """
        aggregated_data = {}
        for key, value in data.items():
            if isinstance(value, list):
                aggregated_data[key + "_mean"] = np.mean(value)
                aggregated_data[key + "_std"] = np.std(value)
            else:
                aggregated_data[key] = value
        return aggregated_data

    def check_keys_valid(self, data: dict[str, float | list[float]]):
        if self.keys is None:
            self.keys = sorted(data.keys())
        else:
            if sorted(data.keys()) != self.keys:
                raise ValueError('Keys of data to log do not match previous keys!')
