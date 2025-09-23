import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(font_scale=2.9)


def main():
    env = 'PointLtl2-v0'
    experiments = [
        'deepltl', 
        'ltl2action', 
        'gcrl-ltl',
        'ours']
    name_mapping = {
        'deepltl': 'DeepLTL', 
        'gcrl-ltl': 'GCRL-LTL', 
        'ltl2action': 'LTL2Action', 
        'deepset': 'DeepLTL', 
        'nocurriculum': 'No curriculum', 
        'deepset_complex': 'DeepLTL',
        "ours": "Ours"}
    df = process_eval_results(env, experiments, name_mapping, smooth_radius=20)
    ci = True

    fig, axes = plt.subplots(1, 2, figsize=(20,8))
    ax = axes[0]
    ax.set(ylabel='Success rate', yticks=np.arange(0, 1.01, 0.1), xlabel='Number of steps', xticks=np.arange(0, 16, 2) * 1000000)
    errorbar = ('ci', 90) if ci else ('sd', 1)
    sns.lineplot(df, x='num_steps', y='success_rate_smooth', errorbar=errorbar, hue='Method', ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)  # remove title='Method'
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    
    ax = axes[1]
    ax.set(ylabel='Violation rate', yticks=np.arange(0, 1.01, 0.1), xlabel='Number of steps', xticks=np.arange(0, 16, 2) * 1000000)
    errorbar = ('ci', 90) if ci else ('sd', 1)
    sns.lineplot(df, x='num_steps', y='violation_rate_smooth', errorbar=errorbar, hue='Method', ax=ax)
    plt.savefig(f'evaluation_curves_{env}.pdf', bbox_inches='tight')
    plt.show()


def process_eval_results(env: str, experiments: list[str], name_mapping=None, smooth_radius=10):
    dfs = []
    for experiment in experiments:
        path = f'eval_results/{env}/{experiment}'
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        for file in files:
            # if experiment == 'super_comp' and (file.startswith('1') or file.startswith('3')):
            #   continue
            df = pd.read_csv(f'{path}/{file}')
            name = name_mapping.get(experiment, experiment)
            df['Method'] = name
            df['seed'] = int(file.split('.')[0])
            for col in ['success_rate', 'violation_rate', 'average_steps', 'return']:
                df[f'{col}_smooth'] = smooth(df[col], smooth_radius)
            dfs.append(df)
        print(f'Loaded {len(files)} files for {name_mapping.get(experiment, experiment)}')
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
