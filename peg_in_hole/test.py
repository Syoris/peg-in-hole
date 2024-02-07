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

from peg_in_hole.utils.neptune import NeptuneTestCallback
from peg_in_hole.tasks.RPL_Insert_3DoF import RPL_Insert_3DoF
from peg_in_hole.rl_algos.rl_algos import get_model, download_model_from_run

# SB3
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

logger = logging.getLogger(__name__)


def test(cfg: DictConfig, run: neptune.Run = None):
    logger.info('##### Testing trained model #####')
    logger.info(f'Task: {cfg.task.name}')

    """ Configs """
    # task_cfg = cfg.task
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

    if cfg.test.model_name is not None:
        model_path, model_type = download_model_from_run(model_path, cfg.test.model_name, cfg)

        model, model_params = get_model(env, cfg.task, model_path, model_type=model_type)
        run['model_params'] = stringify_unsupported(model_params)

    else:
        run['model_params'] = {'algo': 'IK'}

    """ Callbacks """
    neptune_test_callback = NeptuneTestCallback(
        neptune_run=run,
        env_log_freq=cfg.neptune.env_log_freq,
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
        neptune_test_callback.on_step(obs, reward, terminated, truncated, info, action, reset_info)

        if terminated:
            obs, reset_info = env.reset()
    ...
