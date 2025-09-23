#!/usr/bin/env python
import os
import subprocess
import sys
from dataclasses import dataclass
import simple_parsing
import wandb

from utils import kill_all_wandb_processes


@dataclass
class Args:
    script: str
    name: str
    seed: int | list[int]
    device: str
    # cost: float # cost threshold
    num_procs: int = 16
    log_csv: bool = True
    log_wandb: bool = False
    save: bool = True
    steps_per_process: int = 4096
    batch_size: int = 2048
    lr: float = 0.0003
    epochs: int = 80


def main():
    args = simple_parsing.parse(Args)
    env = os.environ.copy()
    env['PYTHONPATH'] = 'src/'
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]
    for seed in seeds:
        command = [
            'python', f'src/train/{args.script}.py',
            '--env', 'PointLtlSafety2-v0',
            '--steps_per_process', str(args.steps_per_process), # '4096',
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--discount', '0.998',
            '--entropy_coef', '0.003',
            # '--cost_limit', str(args.cost),
            '--log_interval', '1',
            '--save_interval', '2', # '2',
            '--epochs', str(args.epochs), # '10',
            '--num_steps', '15_000_000', # '15_000_000',
            '--model_config', 'PointLtlSafety2-v0',
            '--curriculum', 'PointLtlSafety2-v0',
            '--name', args.name,
            '--seed', str(seed),
            '--device', args.device,
            '--num_procs', str(args.num_procs),
        ]
        if args.log_wandb:
            command.append('--log_wandb')
        if not args.log_csv:
            command.append('--no-log_csv')
        if not args.save:
            command.append('--no-save')

        subprocess.run(command, env=env)


if __name__ == '__main__':
    if len(sys.argv) == 1:  # if no arguments are provided, use the following defaults
        sys.argv += '--num_procs 2 --device cpu --name asd --seed 1 --log_csv false --save false'.split(' ')
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted!')
        wandb.finish()
        # kill_all_wandb_processes()
        sys.exit(0)
