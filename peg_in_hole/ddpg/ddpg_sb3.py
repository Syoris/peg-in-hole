import time
import logging

# import tensorflow as tf
import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
import neptune

from peg_in_hole.settings import app_settings

# from peg_in_hole.ddpg.buffer import OUActionNoise, Buffer, update_target
# from peg_in_hole.ddpg.networks import get_actor, get_critic
import peg_in_hole.tasks.RPL_Insert_3DoF  # noqa: F401 Needed to register env to gym
from peg_in_hole.utils.neptune import new_neptune_run

from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import os
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f'Num timesteps: {self.num_timesteps}')
                    print(
                        f'Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}'
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f'Saving new best model to {self.save_path}')
                    self.model.save(self.save_path)

        return True


def initialize_ddpg_model(env, task_cfg):
    lr = task_cfg.rl_hparams.critic_lr
    tau = task_cfg.rl_hparams.tau  # Used to update target networks
    gamma = task_cfg.rl_hparams.buffer.gamma  # Discount factor for future rewards
    buffer_capacity = task_cfg.rl_hparams.buffer.capacity
    batch_size = task_cfg.rl_hparams.buffer.batch_size
    learning_start = 1

    noise_std_dev = task_cfg.rl_hparams.noise_std_dev

    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(env.action_space.shape[-1]),
        sigma=noise_std_dev * np.ones(env.action_space.shape[-1]),
        dt=1e-2,
    )

    model = DDPG(
        'MlpPolicy',
        env,
        action_noise=action_noise,
        verbose=1,
        learning_rate=lr,
        tau=tau,
        gamma=gamma,
        buffer_size=buffer_capacity,
        batch_size=batch_size,
        learning_starts=learning_start,
    )

    return model


def initialize_ppo_model(env, task_cfg):
    model = PPO('MlpPolicy', env, verbose=1)

    return model


def train_ddpg_3dof(cfg: DictConfig):
    logger.info('Starting training of 3dof')

    # General settings
    task_cfg = cfg.task

    # # Neptune logger
    # run = new_neptune_run(neptune_cfg=cfg.neptune)
    # run['task_cfg'] = task_cfg

    # Create the env
    env_name = 'vxUnjamming-v0'
    render_mode = 'human' if cfg.render else None

    # Callback
    # Create log dir
    log_dir = 'tmp/'
    os.makedirs(log_dir, exist_ok=True)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    training_start_time = time.time()
    env = gym.make(env_name, render_mode=render_mode, task_cfg=task_cfg)
    env = Monitor(env, log_dir)

    print(f'init took: {time.time() - training_start_time} sec')

    model = initialize_ddpg_model(env, task_cfg)
    # model = initialize_ppo_model(env, task_cfg)

    n_ep = 100
    n_steps = 250 * n_ep
    model.learn(total_timesteps=n_steps, log_interval=10, progress_bar=True, callback=callback)
    model.save('ddpg_insert')

    vec_env = model.get_env()

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render('human')
        ...


if __name__ == '__main__':
    ...
