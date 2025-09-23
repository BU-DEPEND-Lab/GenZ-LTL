from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str  # name of the experiment
    env: str  # name of the environment
    num_steps: int  # number of steps to train the model for
    seed: int = 0  # random seed
    save_dir: str = 'experiments'  # directory where to save the results
    log_interval: int = 1  # interval at which to log the results (in terms of updates)
    save_interval: int = 2  # interval at which to save the model (in terms of updates)
    eval_interval: int = 200_000  # interval at which to save the model for evaluation (in terms of episodes)
    num_procs: int = 1  # number of processes to use
    device: str = 'cpu'  # device to use for training
    ltl_sampler: str | None = None  # name of the LTL sampler

