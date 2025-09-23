import numpy as np


def average_reward_per_step(returns: list[float], num_frames: list[int]) -> float:
    avgs = []
    if len(returns) != len(num_frames):
        raise ValueError("The length of the returns and num_frames lists must be the same.")
    for i in range(len(returns)):
        avgs.append(returns[i] / num_frames[i])
    return np.mean(avgs)


def average_discounted_return(returns: list[float], num_frames: list[int], discount: float):
    discounted_returns = []
    if len(returns) != len(num_frames):
        raise ValueError("The length of the returns and num_frames lists must be the same.")
    for i in range(len(returns)):
        discounted_returns.append(returns[i] * (discount ** (num_frames[i] - 1)))
    return np.mean(discounted_returns)
