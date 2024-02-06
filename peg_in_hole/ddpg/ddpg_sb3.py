from datetime import datetime
import time
import logging
from pathlib import Path

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
from peg_in_hole.utils.neptune import NeptuneCallback
from peg_in_hole.tasks.RPL_Insert_3DoF import RPL_Insert_3DoF

# SB3
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

logger = logging.getLogger(__name__)


def get_model(env, task_cfg, model_path=None):
    if task_cfg.rl.algo.lower() == 'ddpg':
        model = initialize_ddpg_model(env, task_cfg, model_path)
    elif task_cfg.rl.algo.lower() == 'ppo':
        model = initialize_ppo_model(env, task_cfg, model_path)
    else:
        raise ValueError(f'Model {task_cfg.rl.algo} not recognized')

    return model


def initialize_ddpg_model(env, task_cfg, model_path=None):
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


def initialize_ppo_model(env, task_cfg, model_path=None):
    if model_path is not None:
        model = PPO.load(model_path.as_posix(), env)
    else:
        model = PPO('MlpPolicy', env, verbose=1)

    return model


def train_ddpg_3dof(cfg: DictConfig, run: neptune.Run = None):
    logger.info('Starting training of 3dof')

    """ Configs """
    task_cfg = cfg.task
    run['task_cfg'] = task_cfg
    log_dir = Path(cfg.neptune.temp_save_path) / datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir.mkdir(parents=True, exist_ok=True)

    """ Load past run if needed """
    model_path = None
    if cfg.run_name is not None:
        models = run['model_checkpoints'].fetch()
        latest = max(models.keys(), key=lambda x: int(x))
        model_path = log_dir / f'rl_model_{latest}_steps.zip'

        logger.info(f'Downloading model for {cfg.run_name}. Saving to {model_path.as_posix()}')
        run[f'model_checkpoints/{latest}/model'].download(destination=model_path.as_posix())

    """ Environment """
    render_mode = 'human' if cfg.render else None
    # env = gym.make(env_name, render_mode=render_mode, task_cfg=task_cfg)
    env = RPL_Insert_3DoF(render_mode=render_mode, task_cfg=task_cfg)
    check_env(env)

    env = Monitor(env, log_dir.as_posix())

    """ Model """
    model = get_model(env, task_cfg, model_path)
    total_timesteps_steps = task_cfg.rl.hparams.n_timesteps
    start_timestep = model.num_timesteps
    n_timesteps = total_timesteps_steps - start_timestep

    """ Callbacks """
    save_freq = cfg.neptune.save_freq
    env_log_freq = cfg.neptune.env_log_freq
    log_env = cfg.neptune.log_env

    neptune_callback = NeptuneCallback(
        neptune_run=run,
        log_env=log_env,
        env_log_freq=env_log_freq,
        save_freq=save_freq,
        save_path=log_dir,
        save_replay_buffer=True,
        start_timestep=start_timestep,
    )
    # max_ep_callback = StopTrainingOnMaxEpisodes(max_episodes=n_ep)

    callbacks = [neptune_callback]

    """ Training """
    model.learn(
        total_timesteps=n_timesteps, log_interval=10, progress_bar=True, callback=callbacks, reset_num_timesteps=False
    )
    run.stop()

    """ Test """
    # test_train_ddpg_3dof(env.unwrapped, model)


def test_train_ddpg_3dof(env: RPL_Insert_3DoF, model):
    logger.info('Testing trained model')

    """ Test the trained model """
    vec_env = model.get_env()
    vec_env.render_mode = 'human'

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render('human')
        ...


if __name__ == '__main__':
    ...
