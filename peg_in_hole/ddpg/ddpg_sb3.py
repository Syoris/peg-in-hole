import time
import logging
import os
import matplotlib.pyplot as plt

# import tensorflow as tf
import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
import neptune

# PH
from peg_in_hole.settings import app_settings

# from peg_in_hole.ddpg.buffer import OUActionNoise, Buffer, update_target
# from peg_in_hole.ddpg.networks import get_actor, get_critic
import peg_in_hole.tasks.RPL_Insert_3DoF  # noqa: F401 Needed to register env to gym
from peg_in_hole.utils.neptune import new_neptune_run
from peg_in_hole.tasks.RPL_Insert_3DoF import RPL_Insert_3DoF

# SB3
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.logger import KVWriter

logger = logging.getLogger(__name__)


class NeptuneCallback(BaseCallback):
    def __init__(self, neptune_run: neptune.Run, env_log_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.neptune_run = neptune_run
        self.env_log_freq = env_log_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.n_episodes = 0

        self.episode_log = {
            'step': [],
            'obs': [],
            'command': [],
            'action': [],
            'reward': [],
            'plug_force': [],
            'plug_torque': [],
            'insertion_depth': [],
        }

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        assert 'dones' in self.locals, '`dones` variable is not defined'

        # Record environment data
        self.record_env_step()

        # If episode over, send data to neptune
        episode_done = self.locals.get('dones', None)[0]
        if episode_done:
            # TODO: If multiple envs, need to check the ones that are done and send their data
            self.send_ep_to_neptune()
            self.n_episodes += np.sum(self.locals['dones']).item()

        return True

    def record_env_step(self):
        if self.n_calls % self.env_log_freq == 0:
            obs = self.locals.get('new_obs', None)
            infos = self.locals.get('infos', None)[0]
            command = infos['command']
            plug_force = infos['plug_force']
            plug_torque = infos['plug_torque']
            insertion_depth = infos['insertion_depth']

            action = self.locals.get('actions', None)
            reward = self.locals.get('rewards', None)

            log_dict = {
                'step': self.n_calls,
                'obs': obs,
                'command': command,
                'action': action,
                'reward': reward,
                'plug_force': plug_force,
                'plug_torque': plug_torque,
                'insertion_depth': insertion_depth,
            }

            for param, val in log_dict.items():
                self.episode_log[param].append(val)

    def send_ep_to_neptune(self):
        # Send data to neptune
        ep_logger = self.neptune_run[f'episode/{self.n_episodes}']

        obs = np.vstack(self.episode_log['obs'])
        command = np.vstack(self.episode_log['command'])
        action = np.vstack(self.episode_log['action'])
        reward = np.vstack(self.episode_log['reward'])
        plug_force = np.vstack(self.episode_log['plug_force'])
        plug_torque = np.vstack(self.episode_log['plug_torque'])
        insertion_depth = np.vstack(self.episode_log['insertion_depth'])

        log_dict = {
            'step': self.episode_log['step'],
            'j2_pos': obs[:, 0],
            'j4_pos': obs[:, 1],
            'j6_pos': obs[:, 2],
            'j2_vel': obs[:, 0 + 3],
            'j4_vel': obs[:, 1 + 3],
            'j6_vel': obs[:, 2 + 3],
            'j2_ideal_vel': obs[:, 0 + 6],
            'j4_ideal_vel': obs[:, 1 + 6],
            'j6_ideal_vel': obs[:, 2 + 6],
            'j2_torque': obs[:, 0 + 9],
            'j4_torque': obs[:, 1 + 9],
            'j6_torque': obs[:, 2 + 9],
            'j2_cmd': command[:, 0],
            'j4_cmd': command[:, 1],
            'j6_cmd': command[:, 2],
            'j2_act': action[:, 0],
            'j6_act': action[:, 1],
            'reward': reward[:, 0],
            'plug_force_x': plug_force[:, 0],
            'plug_force_y': plug_force[:, 1],
            'plug_force_z': plug_force[:, 2],
            'plug_torque_x': plug_torque[:, 0],
            'plug_torque_y': plug_torque[:, 1],
            'plug_torque_z': plug_torque[:, 2],
            'insertion_depth_x': insertion_depth[:, 0],
            'insertion_depth_z': insertion_depth[:, 1],
            'insertion_depth_rot': insertion_depth[:, 2],
        }

        for param, val in log_dict.items():
            ep_logger[param].extend(list(val))

        self.episode_log = {
            'step': [],
            'obs': [],
            'command': [],
            'action': [],
            'reward': [],
            'plug_force': [],
            'plug_torque': [],
            'insertion_depth': [],
        }

    def save_model(self):
        ...
        # Save?
        # if self.n_calls % self.check_freq == 0:
        #     # Retrieve training reward
        #     x, y = ts2xy(load_results(self.log_dir), 'timesteps')
        #     if len(x) > 0:
        #         # Mean training reward over the last 100 episodes
        #         mean_reward = np.mean(y[-100:])
        #         if self.verbose >= 1:
        #             print(f'Num timesteps: {self.num_timesteps}')
        #             print(
        #                 f'Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}'
        #             )

        #         # New best model, you could save the agent here
        #         if mean_reward > self.best_mean_reward:
        #             self.best_mean_reward = mean_reward
        #             # Example for saving best model
        #             if self.verbose >= 1:
        #                 print(f'Saving new best model to {self.save_path}')
        #             self.model.save(self.save_path)


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

        # Parameters
        # self.training_env.actions

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

    # Neptune logger
    run = new_neptune_run(neptune_cfg=cfg.neptune)
    run['task_cfg'] = task_cfg

    # Create the env
    # env_name = 'vxUnjamming-v0'
    render_mode = 'human' if cfg.render else None

    """ Callbacks """
    # Create log dir
    log_dir = 'tmp/'
    os.makedirs(log_dir, exist_ok=True)
    save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    neptune_callback = NeptuneCallback(neptune_run=run, env_log_freq=5, log_dir=log_dir)
    callbacks = [neptune_callback]

    """ Environment """
    training_start_time = time.time()
    # env = gym.make(env_name, render_mode=render_mode, task_cfg=task_cfg)
    env = RPL_Insert_3DoF(render_mode=render_mode, task_cfg=task_cfg)
    check_env(env)

    env = Monitor(env, log_dir)

    print(f'init took: {time.time() - training_start_time} sec')

    model = initialize_ddpg_model(env, task_cfg)
    # model = initialize_ppo_model(env, task_cfg)

    n_ep = 100
    n_steps = 250 * n_ep
    model.learn(total_timesteps=n_steps, log_interval=10, progress_bar=True, callback=callbacks)
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
