import argparse
from colorama import Fore

from utils.logging.logger import Logger


class TextLogger(Logger):
    """
    A logger that logs to standard output.
    """

    def __init__(self, config: argparse.Namespace):
        super().__init__(config)

    def log_config(self):
        print(self.config.experiment)

    def log(self, data: dict[str, float | list[float]]):
        data = self.aggregate(data)
        self.check_keys_valid(data)
        row = ''
        for key, value in data.items():
            if key == 'avg_goal_success':
                continue
            short_name = self.get_short_name(key)
            row += f'{short_name}: '
            if isinstance(value, float):
                row += f'{value:.2f} | '
            else:
                row += f'{value} | '
        row = row[:-3]  # remove trailing ' | '
        print(row)

    @staticmethod
    def get_short_name(key: str) -> str:
        if key == 'return_per_episode_mean':
            return 'rμ'
        elif key == 'return_per_episode_std':
            return 'rσ'
        elif key == 'num_steps_per_episode_mean':
            return 'sμ'
        elif key == 'num_steps_per_episode_std':
            return 'sσ'
        elif key == 'success_per_episode_mean':
            return 'Pμ'
        elif key == 'success_per_episode_std':
            return 'Pσ'
        elif key == 'violation_per_episode_mean':
            return 'Vμ'
        elif key == 'violation_per_episode_std':
            return 'Vσ'
        elif key == 'duration':
            return 't'
        else:
            return key

    @staticmethod
    def info(message: str):
        print(message)

    @staticmethod
    def important_info(message: str):
        print(f'{Fore.GREEN}{message}{Fore.RESET}')
