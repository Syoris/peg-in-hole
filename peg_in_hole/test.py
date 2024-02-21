from datetime import datetime
import logging
from pathlib import Path

# import tensorflow as tf
import numpy as np
from omegaconf import DictConfig
import neptune
from neptune.utils import stringify_unsupported

# PH
from peg_in_hole.settings import app_settings

from peg_in_hole.utils.neptune import NeptuneTestCallback, init_neptune_run
from peg_in_hole.tasks.RPL_Insert_3DoF import RPL_Insert_3DoF
from peg_in_hole.rl_algos.rl_algos import get_model, download_model_from_run

# SB3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

logger = logging.getLogger(__name__)


def test(cfg: DictConfig):
    logger.info('##### Testing trained model #####')
    logger.info(f'Task: {cfg.task.name}')

    if cfg.neptune.use_neptune is not None:
        run = init_neptune_run(cfg.train.run_name, neptune_cfg=cfg.neptune)
    else:
        run = None

    """ Configs """
    if run is not None:
        run['cfg'] = stringify_unsupported(cfg)
        run['sys/tags'].add('test')

    log_dir = Path(cfg.neptune.temp_save_path) / datetime.today().strftime('%Y-%m-%d_%H-%M-%S_test')
    log_dir.mkdir(parents=True, exist_ok=True)

    """ Environment """
    render_mode = 'human' if cfg.test.render else None
    env = RPL_Insert_3DoF(render_mode=render_mode, task_cfg=cfg.task)
    check_env(env)

    """ Model """
    model_path = log_dir
    model = None

    model_params_for_run = None
    if cfg.test.model_name is not None:
        model_path, model_type = download_model_from_run(model_path, cfg.test.model_name, cfg)

        model, model_params = get_model(env, cfg.task, model_path, model_type=model_type)
        model_params_for_run = stringify_unsupported(model_params)

    else:
        model_params_for_run = {'algo': 'IK'}

    if run is not None:
        run['model_params'] = model_params_for_run

    """ Callbacks """
    if run is not None:
        neptune_test_callback = NeptuneTestCallback(
            neptune_run=run,
            env_log_freq=cfg.test.log_freq,
        )

    """ Test the trained model """
    obs, reset_info = env.reset()
    n_epochs = cfg.test.n_epochs
    n_steps = n_epochs * 250

    for _ in range(n_steps):
        if model is not None:
            action, _states = model.predict(obs)
            # action = env.action_space.sample()  # Random action
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)  # IK Only

        obs, reward, terminated, truncated, info = env.step(action)
        if run is not None:
            neptune_test_callback.on_step(obs, reward, terminated, truncated, info, action, reset_info)

        if terminated:
            obs, reset_info = env.reset()
    ...