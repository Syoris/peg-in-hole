import logging

# import tensorflow as tf
import numpy as np

# PH
from peg_in_hole.settings import app_settings

# SB3
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

logger = logging.getLogger(__name__)


def get_model(env, task_cfg, model_path=None):
    if task_cfg.rl.algo.lower() == 'ddpg':
        model, model_params = initialize_ddpg_model(env, task_cfg, model_path)
    elif task_cfg.rl.algo.lower() == 'ppo':
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

    model_params = {}

    return model, model_params


def initialize_ppo_model(env, task_cfg, model_path=None):
    if model_path is not None:
        model = PPO.load(model_path.as_posix(), env)
    else:
        model = PPO('MlpPolicy', env, verbose=1)

    model_params = {
        'bath_size': model.batch_size,
        'gamma': model.gamma,
        'learning_rate': model.learning_rate,
        'gae_lambda': model.gae_lambda,
        'seed': model.seed,
        'vf_coef': model.vf_coef,
        'policy': model.policy_class.__name__,
    }

    return model, model_params
