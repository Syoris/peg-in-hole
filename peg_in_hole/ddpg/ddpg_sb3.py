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

logger = logging.getLogger(__name__)


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

    training_start_time = time.time()
    env = gym.make(env_name, render_mode=render_mode, task_cfg=task_cfg)
    print(f'init took: {time.time() - training_start_time} sec')

    # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions), dt=1e-2)

    # model = PPO('MlpPolicy', env, verbose=1)
    model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)

    n_ep = 100
    n_steps = 250 * n_ep
    model.learn(total_timesteps=n_steps, log_interval=10, progress_bar=True)
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
