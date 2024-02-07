import logging
import re
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

# PH
from peg_in_hole.utils.neptune import init_neptune_run

# SB3
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

logger = logging.getLogger(__name__)


def get_model(env, task_cfg, model_path=None, model_type=None):
    """To initialize the model based on the task config. If model_type is specified, intialize that model.
    Else, initialize the model from task_cfg.

    Args:
        env (_type_): _description_
        task_cfg (_type_): _description_
        model_path (_type_, optional): _description_. Defaults to None.
        model_type (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if model_type is None:
        model_type = task_cfg.rl.algo

    if model_type.lower() == 'ddpg':
        model, model_params = initialize_ddpg_model(env, task_cfg, model_path)
    elif model_type.lower() == 'ppo':
        model, model_params = initialize_ppo_model(env, task_cfg, model_path)
    else:
        raise ValueError(f'Model {task_cfg.rl.algo} not recognized')

    return model, model_params


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

    model_params = {
        'algo': 'DDPG',
    }

    return model, model_params


def initialize_ppo_model(env, task_cfg, model_path=None):
    if model_path is not None:
        model = PPO.load(model_path.as_posix(), env)
    else:
        model = PPO('MlpPolicy', env, verbose=1)

    model_params = {
        'algo': 'PPO',
        'bath_size': model.batch_size,
        'gamma': model.gamma,
        'learning_rate': model.learning_rate,
        'gae_lambda': model.gae_lambda,
        'seed': model.seed,
        'vf_coef': model.vf_coef,
        'policy': model.policy_class.__name__,
    }

    return model, model_params


def download_model_from_run(model_path: Path, run_name: str, cfg: DictConfig) -> Path:
    """To download a model from a neptune run.

    Args:
        model_path (Path): Where to save the model
        run_name (str):
            - 'PH-XX_YYYY' where XX is the run id and YYYY is the model timesteps, OR
            - 'PH-XX' where XX is the run id and the latest model will be downloaded

    Returns:
        Path: Path to the downloaded model
    """
    if re.match(r'^PH-\d+$', run_name) is not None:
        run_id = run_name
        model_ts = None

    elif re.match(r'^PH-\d+_\d+$', run_name) is not None:
        run_id, model_ts = run_name.split('_')

    else:
        raise ValueError(f'Invalid run name: {run_name}')

    model_run = init_neptune_run(run_id, neptune_cfg=cfg.neptune, read_only=True)

    if model_ts is None:
        models = model_run['model_checkpoints'].fetch()
        model_ts = max(models.keys(), key=lambda x: int(x))

    model_path = model_path / f'rl_model_{model_ts}_steps.zip'

    logger.info(f'Downloading model for {model_run}. Saving to {model_path.as_posix()}')
    model_run[f'model_checkpoints/{model_ts}/model'].download(destination=model_path.as_posix())

    model_type = model_run['cfg/task/rl/algo'].fetch()

    model_run.stop()

    return model_path, model_type
