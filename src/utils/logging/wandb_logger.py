import argparse
import os

import wandb

import utils
from utils.logging.logger import Logger


class WandbLogger(Logger):
    """
    A logger that logs to Weights & Biases.
    """

    WANDB_FILE_NAME = 'wandb_id.txt'

    def __init__(self, config: argparse.Namespace, project_name: str, resuming: bool = False):
        super().__init__(config)
        self.project_name = project_name
        self.run_id = None
        if resuming:
            wandb_id_file = f'{utils.get_experiment_path_from_config(config)}/{self.WANDB_FILE_NAME}'
            if not os.path.exists(wandb_id_file):
                raise FileNotFoundError(f'Trying to resume, but no wandb_id.txt file found in {wandb_id_file}.')
            with open(wandb_id_file, 'r') as f:
                self.run_id = f.read().strip()

    def log_config(self):
        if self.run_id is not None:
            wandb.init(
                project=self.project_name,
                id=self.run_id,
                resume='must',
            )
        else:
            run = wandb.init(
                project=self.project_name,
                config=vars(self.config),
            )
            wandb_id_file = f'{utils.get_experiment_path_from_config(self.config)}/{self.WANDB_FILE_NAME}'
            with open(wandb_id_file, 'w') as f:
                f.write(run.id)

    def log(self, data: dict[str, float | list[float]]):
        data = self.aggregate(data)
        self.check_keys_valid(data)
        del data['avg_goal_success']
        wandb.log(data)

    def finish(self):
        wandb.finish()
