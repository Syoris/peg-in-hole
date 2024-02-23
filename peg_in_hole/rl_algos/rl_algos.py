import logging
import re
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

# PH
from peg_in_hole.utils.neptune import init_neptune_run

# SB3
from stable_baselines3 import DDPG, PPO, TD3
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

    model_type = model_type.lower()
    init_func = RL_ALGOS.get(model_type, None)

    if init_func is None:
        raise ValueError(f'Model {task_cfg.rl.algo} not recognized')

    model, model_params = init_func(env, task_cfg, model_path)

    return model, model_params


def initialize_ddpg_model(env, task_cfg, model_path=None):
    if model_path is not None:
        if 'test' in model_path.as_posix():
            logger.info(f'Loading DDPG model from {model_path.as_posix()}')
            model = DDPG.load(model_path.as_posix(), env)

        else:
            raise ValueError('DDPG model not supported for training yet')

    else:
        ddpg_params = task_cfg.rl.hparams.ddpg
        lr = ddpg_params.lr
        tau = ddpg_params.tau  # Used to update target networks
        gamma = ddpg_params.buffer.gamma  # Discount factor for future rewards
        buffer_capacity = ddpg_params.buffer.capacity
        batch_size = ddpg_params.buffer.batch_size
        learning_start = 1

        noise_std_dev = ddpg_params.noise_std_dev

        # action_noise = OrnsteinUhlenbeckActionNoise(
        #     mean=np.zeros(env.action_space.shape[-1]),
        #     sigma=noise_std_dev * np.ones(env.action_space.shape[-1]),
        #     dt=1e-2,
        # )

        # action_noise = OrnsteinUhlenbeckActionNoise(
        #     mean=np.zeros(env.action_space.shape[-1]),
        #     sigma=0.1 * np.ones(env.action_space.shape[-1]),
        # )

        action_noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[-1]), sigma=0.5 * np.ones(env.action_space.shape[-1])
        )

        model = DDPG(
            'MlpPolicy',
            env,
            action_noise=action_noise,
            verbose=1,
            # learning_rate=lr,
            # tau=tau,
            # gamma=gamma,
            # buffer_size=buffer_capacity,
            # batch_size=batch_size,
            # learning_starts=learning_start,
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


def initialize_td3_model(env, task_cfg, model_path=None):
    if model_path is not None:
        if 'test' in model_path.as_posix():
            logger.info(f'Loading TD3 model from {model_path.as_posix()}')
            model = TD3.load(model_path.as_posix(), env)

        else:
            raise ValueError('TD3 model not supported for training yet')  # TODO: Implement training for TD3

    else:
        td3_params = task_cfg.rl.hparams.td3
        # lr = ddpg_params.lr
        # tau = ddpg_params.tau  # Used to update target networks
        # gamma = ddpg_params.buffer.gamma  # Discount factor for future rewards
        # buffer_capacity = ddpg_params.buffer.capacity
        # batch_size = ddpg_params.buffer.batch_size
        # learning_start = 1

        # noise_std_dev = ddpg_params.noise_std_dev

        action_noise = NormalActionNoise(
            mean=np.zeros(env.action_space.shape[-1]), sigma=0.1 * np.ones(env.action_space.shape[-1])
        )

        model = TD3(
            'MlpPolicy',
            env,
            action_noise=action_noise,
            verbose=1,
            # learning_rate=lr,
            # tau=tau,
            # gamma=gamma,
            # buffer_size=buffer_capacity,
            # batch_size=batch_size,
            # learning_starts=learning_start,
        )

    model_params = {
        'algo': 'TD3',
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

    logger.info(f'Downloading model for {model_run["sys/id"].fetch()}. Saving to {model_path.as_posix()}')
    model_run[f'model_checkpoints/{model_ts}/model'].download(destination=model_path.as_posix())

    model_type = model_run['cfg/task/rl/algo'].fetch()

    model_run.stop()

    return model_path, model_type


RL_ALGOS = {
    'ddpg': initialize_ddpg_model,
    'ppo': initialize_ppo_model,
    'td3': initialize_td3_model,
}
