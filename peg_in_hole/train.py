from datetime import datetime
import time
import logging
from pathlib import Path

# import tensorflow as tf
import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
import neptune
from neptune.utils import stringify_unsupported

# PH
from peg_in_hole.settings import app_settings

# from peg_in_hole.ddpg.buffer import OUActionNoise, Buffer, update_target
# from peg_in_hole.ddpg.networks import get_actor, get_critic
import peg_in_hole.tasks.RPL_Insert_3DoF  # noqa: F401 Needed to register env to gym
from peg_in_hole.utils.neptune import NeptuneTrainCallback, init_neptune_run
from peg_in_hole.tasks.RPL_Insert_3DoF import RPL_Insert_3DoF
from peg_in_hole.rl_algos.rl_algos import get_model

# SB3
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

logger = logging.getLogger(__name__)


def train(cfg: DictConfig):
    logger.info('Starting training of 3dof')
    logger.info('##### Training #####')
    logger.info(f'Task: {cfg.task.name}')

    if cfg.train.use_neptune is not False:
        run = init_neptune_run(cfg.train.run_name, neptune_cfg=cfg.neptune)
    else:
        run = None

    """ Configs """
    task_cfg = cfg.task
    if run is not None:
        run['cfg'] = stringify_unsupported(cfg)
        run['sys/tags'].add('train')

    log_dir = Path(cfg.neptune.temp_save_path) / datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir.mkdir(parents=True, exist_ok=True)

    """ Load past run if needed """
    model_path = None
    if cfg.train.run_name is not None and run is not None:
        models = run['model_checkpoints'].fetch()
        latest = max(models.keys(), key=lambda x: int(x))
        model_path = log_dir / f'rl_model_{latest}_steps.zip'

        logger.info(f'Downloading model for {cfg.train.run_name}. Saving to {model_path.as_posix()}')
        run[f'model_checkpoints/{latest}/model'].download(destination=model_path.as_posix())

    """ Environment """
    # TODO: Create env based on config
    render_mode = 'human' if cfg.train.render else None
    # env = gym.make(env_name, render_mode=render_mode, task_cfg=task_cfg)
    env = RPL_Insert_3DoF(render_mode=render_mode, task_cfg=task_cfg)
    check_env(env)

    env = Monitor(env, log_dir.as_posix())

    """ Model """
    model, model_params = get_model(env, task_cfg, model_path)
    if run is not None:
        run['model_params'] = stringify_unsupported(model_params)

    total_timesteps_steps = task_cfg.rl.hparams.n_timesteps
    start_timestep = model.num_timesteps
    n_timesteps = total_timesteps_steps - start_timestep
    logger.info(f'RL Algo parameters:\n{model_params}')

    """ Callbacks """
    save_freq = cfg.neptune.save_freq
    env_log_freq = cfg.neptune.env_log_freq
    log_env = cfg.neptune.log_env

    if run is not None:
        neptune_callback = NeptuneTrainCallback(
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
    else:
        callbacks = []

    """ Training """
    model.learn(
        total_timesteps=n_timesteps, log_interval=500, progress_bar=True, callback=callbacks, reset_num_timesteps=False
    )

    if run is not None:
        run.stop()


if __name__ == '__main__':
    ...
