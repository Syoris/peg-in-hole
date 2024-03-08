"""
To make plots from data saved in Neptune.

See `kinova-gen2-unjamming\3DOF\RL_setup\ddpg_makeplots.py` for examples
"""

import hydra
from omegaconf import DictConfig
import neptune

import h5py
import numpy as np
import random
import math
import matplotlib.pyplot as plt

from peg_in_hole.utils.neptune import init_neptune_run
from peg_in_hole.settings import app_settings

SAVE_PATH = app_settings.data_path / 'plots'

tex_fonts = {
    # Use LaTeX to write all text
    'text.usetex': True,
    'font.family': 'serif',
    # Use 10pt font in plots, to match 10pt font in document
    'axes.labelsize': 14,
    'font.size': 14,
    # Make the legend/label fonts a little smaller
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
}
plt.rcParams.update(tex_fonts)


@hydra.main(version_base=None, config_name='config', config_path='../cfg')
def make_plots(cfg: DictConfig):
    print('------------- Making plots -------------')

    runs_list = [248, 202, 203]

    # Load the data
    runs_data_dict = {}
    for each_run in runs_list:
        run_data = load_data_from_neptune(each_run, cfg.neptune)
        runs_data_dict[each_run] = run_data

    # Plots
    training_reward_plot_combined(runs_data_dict)
    # training_reward_plot_subplot(runs_data_dict)

    print('Done')


def load_data_from_neptune(run_id: int, neptune_cfg: DictConfig):
    data_dict = {}

    # Load the data
    run: neptune.Run = init_neptune_run(run_id, neptune_cfg=neptune_cfg, read_only=True)

    # Metadata
    data_dict['reward_func'] = run['env_params/reward_function'].fetch()

    # Data
    fields = ['ep_reward', 'mean_reward']

    for each_field in fields:
        data_dict[each_field] = run[f'data/{each_field}'].fetch_values()

    # Close the run
    run.stop()

    return data_dict


def training_reward_plot_combined(runs_data: dict, save_plot=True):
    print('Plotting `training_reward_plot_combined`...')

    fig, ax = plt.subplots()
    # fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, constrained_layout=True, figsize=(7,7))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if you have more than 7 runs

    # Plot the data
    # Loop to plot all runs, doesn't work for more than two plots?
    # for i, (run_id, run_data) in enumerate(runs_data.items()):
    #     color = colors[i % len(colors)]
    #     ax_i = ax if i == 0 else ax.twinx()
    #     ax_i.plot(run_data['ep_reward'].value, color=color, label=f'Run {run_id}')
    #     ax_i.set_ylabel(f'Reward {run_id}', color=color)
    #     ax_i.tick_params(axis='y', colors=color)

    run_id = 202
    run_color = 'b'
    rew_func = runs_data[run_id]['reward_func']
    run_reward_vals = runs_data[run_id]['mean_reward'].value
    ax.plot(run_reward_vals, label=rew_func, color=run_color)
    ax.set_ylabel(f'{rew_func}', color=run_color)
    # ax.tick_params(axis='y', labelcolor=run_color)
    ax.minorticks_on()
    ax.grid(True)  # Toggle grid on

    ax2 = ax.twinx()
    run_id = 203
    run_color = 'g'
    rew_func = runs_data[run_id]['reward_func']
    run_reward_vals = runs_data[run_id]['mean_reward'].value
    ax2.plot(run_reward_vals, label=rew_func, color=run_color)
    ax2.set_ylabel(f'{rew_func}', color=run_color)
    # ax2.tick_params(axis='y', labelcolor=run_color)
    ax2.minorticks_on()
    ax2.grid(True)  # Toggle grid on

    # ax3 = ax.twinx()
    # run_id = 248
    # run_color = 'g'
    # rew_func = runs_data[run_id]['reward_func']
    # run_reward_vals = runs_data[run_id]['mean_reward'].value
    # ax3.plot(run_reward_vals, label=rew_func, color=run_color)
    # ax3.set_ylabel(f'Reward {run_id}', color=run_color)
    # ax3.tick_params(axis='y', labelcolor=run_color)

    # Format the plot
    # fig.tight_layout()
    fig.legend(loc='lower right')
    # ax.legend(loc='best')
    # plt.legend().get_frame().set_linewidth(0.0)

    # ax.set_ylabel('Reward')
    ax.set_xlabel('Training time step')

    ax.grid(True)  # Toggle grid on

    # Save the plot
    if save_plot:
        plt.savefig(SAVE_PATH / 'reward_func_training_comp_plot.png', bbox_inches='tight')

    plt.show()
    # plt.close()


def training_reward_plot_subplot(runs_data: dict, save_plot=True):
    print('Plotting `training_reward_plot_subplot`...')

    n_plots = len(runs_data)

    fig, axes = plt.subplots(n_plots, 1, sharex=True, constrained_layout=True, figsize=(7, 7))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if you have more than 7 runs

    # Plot the data
    for i, (run_id, run_data) in enumerate(runs_data.items()):
        color = colors[i % len(colors)]
        rew_func = run_data['reward_func']
        run_reward_vals = run_data['mean_reward'].value

        ax = axes[i]
        ax.plot(run_reward_vals, color=color, label=f'{rew_func}')
        ax.set_ylabel(f'{rew_func}')
        # ax.tick_params(axis='y', colors=color)
        ax.grid(True)  # Toggle grid on
        ax.minorticks_on()

    # Format the plot
    fig.tight_layout()
    # fig.legend(loc='lower right')
    # ax.legend()
    # plt.legend().get_frame().set_linewidth(0.0)

    ax.set_xlabel('Time step')  # Set the x-axis label for the last plot

    # Save the plot
    if save_plot:
        plt.savefig(SAVE_PATH / 'reward_func_training_comp_subs_plot.png', bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    make_plots()
