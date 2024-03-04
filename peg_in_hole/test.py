from datetime import datetime, timedelta
import time
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

# from tqdm import tqdm
from tqdm.rich import tqdm

logger = logging.getLogger(__name__)


def test(cfg: DictConfig):
    logger.info('##### Testing trained model #####')
    logger.info(f'Task: {cfg.task.name}')

    if cfg.test.use_neptune is not False:
        run = init_neptune_run(cfg.train.run_name, neptune_cfg=cfg.neptune)
    else:
        run = None

    """ Configs """
    # TODO: Load config from model if provided. Is it necessary?
    if run is not None:
        run['cfg'] = stringify_unsupported(cfg)
        run['sys/tags'].add('test')

    log_dir = Path(cfg.neptune.temp_save_path) / datetime.today().strftime('%Y-%m-%d_%H-%M-%S_test')
    log_dir.mkdir(parents=True, exist_ok=True)

    """ Environment """
    render_mode = 'human' if cfg.test.render else None
    env = RPL_Insert_3DoF(render_mode=render_mode, task_cfg=cfg.task)
    # check_env(env)

    # Env parameters
    if run is not None:
        run['env_params'] = stringify_unsupported(env.unwrapped.get_params())

    """ Model """
    model_path = log_dir
    model = None

    model_params_for_run = None
    if cfg.test.model_id is not None and cfg.test.model_id != 'None':
        model_path, model_type, model_url = download_model_from_run(model_path, str(cfg.test.model_id), cfg)

        model, model_params = get_model(env, cfg.task, model_path, model_type=model_type)
        model_params_for_run = stringify_unsupported(model_params)
        model_params_for_run['model'] = f'PH-{cfg.test.model_id}'
        model_params_for_run['model_url'] = model_url

    else:
        model_params_for_run = {'algo': 'IK', 'model': 'None'}

    if run is not None:
        run['model_params'] = model_params_for_run

    """ Callbacks """
    if run is not None:
        neptune_test_callback = NeptuneTestCallback(
            neptune_run=run,
            env_log_freq=cfg.test.log_freq,
        )

    """ Rendering speed """
    render_speed = cfg.test.render_speed
    ts = cfg.task.env.h
    dt = ts / render_speed

    """ Test the trained model """
    obs, reset_info = env.reset()
    epoch_num = 0
    n_epochs = cfg.test.n_epochs
    n_steps = n_epochs * 250

    for _ in tqdm(range(n_steps)):
        step_start_time = time.time()

        if model is not None:
            action, _states = model.predict(obs, deterministic=True)
            # action = env.action_space.sample()  # Random action
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)  # IK Only

        obs, reward, terminated, truncated, info = env.step(action)
        if run is not None:
            neptune_test_callback._on_step(obs, reward, terminated, truncated, info, action, reset_info)

        if terminated or truncated:
            obs, reset_info = env.reset()
            epoch_num += 1
            if epoch_num >= n_epochs:
                break

        # Wait for time period
        while (time.time() - step_start_time) < dt and render_mode is not None:
            time.sleep(0.001)

    if run is not None:
        neptune_test_callback._on_test_end()
        run.stop()
