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

from settings import app_settings
import logging
import traceback
import hydra
from omegaconf import DictConfig

from peg_in_hole.train import train
from peg_in_hole.test import test


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name='config', config_path='../cfg')
def ik_test(cfg: DictConfig):
    """Environment"""
    render_mode = 'human' if cfg.test.render else None
    env = RPL_Insert_3DoF(render_mode=render_mode, task_cfg=cfg.task)
    check_env(env)

    """ Run """
    render_speed = 1  # 1 is normal speed, 0.5 is half speed, 2 is double speed
    ts = cfg.task.env.h
    dt = ts / render_speed

    obs, reset_info = env.reset()

    while True:
        step_start_time = time.time()

        action = np.array([0.0, 0.0], dtype=np.float32)  # IK Only

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            obs, reset_info = env.reset()

        # Wait for time period
        while (time.time() - step_start_time) < dt and render_mode is not None:
            time.sleep(0.001)


if __name__ == '__main__':
    ik_test()
