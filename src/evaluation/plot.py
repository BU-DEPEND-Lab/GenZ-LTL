import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(font_scale=.8)


def main():
    env = 'PointLtl2-v0'
    experiments = ['evaluation']
    name_mapping = {'pre': 'pretraining', 'cat': 'standard'}
    df = process_logs(env, experiments, name_mapping)
    ax = sns.relplot(df, x='num_steps', y='return_smooth', kind='line', errorbar='sd', hue='experiment')
    ax.set(ylabel='success rate')
    # plt.savefig(os.path.expanduser('~/tmp/plot.png'))
    plt.show()


def process_logs(env: str, experiments: list[str], name_mapping=None, smooth_radius=10):
    if name_mapping is None:
        name_mapping = dict()
    dfs = []
    for experiment in experiments:
        path = f'experiments/ppo/{env}/{experiment}'
        seeds = [int(x) for x in os.listdir(path) if os.path.isdir(f'{path}/{x}') and str.isnumeric(x)]
        for seed in seeds:
            df = pd.read_csv(f'{path}/{seed}/log.csv')
            name = name_mapping.get(experiment, experiment)
            df['experiment'] = name
            df['return'] = df['return_per_episode_mean']
            df['seed'] = seed
            for col in ['return', 'adr', 'arps']:
                df[f'{col}_smooth'] = smooth(df[col], smooth_radius)
            dfs.append(df)
    result = pd.concat(dfs)
    if result.isna().any().any():
        print('Warning: data contains NaN values')
    return result


def smooth(row, radius):
    """
    Computes the moving average over the given row of data. Returns an array of the same shape as the original row.
    """
    y = np.ones(radius)
    z = np.ones(len(row))
    return np.convolve(row, y, 'same') / np.convolve(z, y, 'same')


if __name__ == '__main__':
    main()
