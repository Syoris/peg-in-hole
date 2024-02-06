import time
import logging
import tensorflow as tf
import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
import neptune

from peg_in_hole.settings import app_settings
from peg_in_hole.ddpg.buffer import OUActionNoise, Buffer, update_target
from peg_in_hole.ddpg.networks import get_actor, get_critic
import peg_in_hole.tasks.RPL_Insert_3DoF  # noqa: F401 Needed to register env to gym
from peg_in_hole.utils.neptune import init_neptune_run


logger = logging.getLogger(__name__)


def log_episode_data(neptune_run: neptune.Run, episode_data: dict, episode: int) -> dict:
    ep_logger = neptune_run[f'episode/{episode}']

    obs = np.vstack(episode_data['obs'])
    command = np.vstack(episode_data['command'])
    action = np.vstack(episode_data['action'])
    reward = np.vstack(episode_data['reward'])
    plug_force = np.vstack(episode_data['plug_force'])
    plug_torque = np.vstack(episode_data['plug_torque'])
    insertion_depth = np.vstack(episode_data['insertion_depth'])

    log_dict = {
        'step': episode_data['step'],
        'j2_pos': obs[:, 0],
        'j4_pos': obs[:, 1],
        'j6_pos': obs[:, 2],
        'j2_vel': obs[:, 0 + 3],
        'j4_vel': obs[:, 1 + 3],
        'j6_vel': obs[:, 2 + 3],
        'j2_ideal_vel': obs[:, 0 + 6],
        'j4_ideal_vel': obs[:, 1 + 6],
        'j6_ideal_vel': obs[:, 2 + 6],
        'j2_torque': obs[:, 0 + 9],
        'j4_torque': obs[:, 1 + 9],
        'j6_torque': obs[:, 2 + 9],
        'j2_cmd': command[:, 0],
        'j4_cmd': command[:, 1],
        'j6_cmd': command[:, 2],
        'j2_act': action[:, 0],
        'j6_act': action[:, 1],
        'reward': reward[:, 0],
        'plug_force_x': plug_force[:, 0],
        'plug_force_y': plug_force[:, 1],
        'plug_force_z': plug_force[:, 2],
        'plug_torque_x': plug_torque[:, 0],
        'plug_torque_y': plug_torque[:, 1],
        'plug_torque_z': plug_torque[:, 2],
        'insertion_depth_x': insertion_depth[:, 0],
        'insertion_depth_z': insertion_depth[:, 1],
        'insertion_depth_rot': insertion_depth[:, 2],
    }

    for param, val in log_dict.items():
        ep_logger[param].extend(list(val))

    episode_log = {
        'step': [],
        'obs': [],
        'command': [],
        'action': [],
        'reward': [],
        'plug_force': [],
        'plug_torque': [],
        'insertion_depth': [],
    }

    return episode_log


def policy(state, noise_object, actor_model, lower_bound, upper_bound):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()

    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return np.squeeze(legal_action)


def train3dof(cfg: DictConfig):
    logger.info('Starting training of 3dof')

    # General settings
    task_cfg = cfg.task

    # Neptune logger
    run = init_neptune_run(neptune_cfg=cfg.neptune)
    run['task_cfg'] = task_cfg

    # Create the env
    env_name = 'vxUnjamming-v0'
    render_mode = 'human' if cfg.render else None

    training_start_time = time.time()
    env = gym.make(env_name, render_mode=render_mode, task_cfg=task_cfg)
    print(f'init took: {time.time() - training_start_time} sec')

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    logger.info(f'Size of State Space: {num_states}')
    logger.info(f'Size of Action Space: {num_states}')

    upper_bound = env.action_space.high
    lower_bound = env.action_space.low
    logger.info(f'Max Value of Action: {upper_bound}')
    logger.info(f'Min Value of Action: {lower_bound}')

    # Networks
    noise_std_dev = task_cfg.rl_hparams.noise_std_dev
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(noise_std_dev) * np.ones(1))

    actor_model = get_actor(num_states=num_states, num_actions=num_actions)
    critic_model = get_critic(num_states=num_states, num_actions=num_actions)

    actor_target = get_actor(num_states=num_states, num_actions=num_actions)
    critic_target = get_critic(num_states=num_states, num_actions=num_actions)

    # Making the weights equal initially
    actor_target.set_weights(actor_model.get_weights())
    critic_target.set_weights(critic_model.get_weights())

    # Learning rate for actor-critic models
    critic_lr = task_cfg.rl_hparams.critic_lr
    actor_lr = task_cfg.rl_hparams.actor_lr

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    total_episodes = task_cfg.rl_hparams.episodes

    tau = task_cfg.rl_hparams.tau  # Used to update target networks

    buffer = Buffer(
        num_states=num_states,
        num_actions=num_actions,
        actor_model=actor_model,
        target_actor=actor_target,
        actor_optimizer=actor_optimizer,
        critic_model=critic_model,
        target_critic=critic_target,
        critic_optimizer=critic_optimizer,
        gamma=task_cfg.rl_hparams.buffer.gamma,  # Discount factor for future rewards
        buffer_capacity=task_cfg.rl_hparams.buffer.capacity,
        batch_size=task_cfg.rl_hparams.buffer.batch_size,
    )

    # To store reward history of each episode
    logging_freq = task_cfg.general.log_freq

    # Save here the states at each log_freq
    episode_log = {
        'step': [],
        'obs': [],
        'command': [],
        'action': [],
        'reward': [],
        'plug_force': [],
        'plug_torque': [],
        'insertion_depth': [],
    }

    ep_reward_list = []
    ep_force_list = []
    ep_torque_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    avg_force_list = []
    end_depth_list = []
    avg_torque_list = []
    avg_height_list = []

    training_start_time = time.time()

    # Training
    for episode_count in range(total_episodes):
        print(f'--------- Episode {episode_count} ---------')
        prev_state, _ = env.reset()
        # prev_state = prev_state[0]
        episodic_reward = 0
        episodic_force = 0
        episodic_torque = 0

        step_count = 0
        while True:
            # TODO: Move to function run_episode()
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state, ou_noise, actor_model, lower_bound, upper_bound)

            # Recieve state and reward from environment.
            obs, reward, done, truncated, info = env.step(action)

            plug_force = info['plug_force']
            plug_torque = info['plug_torque']
            command = info['command']
            insertion_depth = info['insertion_depth']

            force_norm = np.linalg.norm(plug_force)
            torque_norm = np.linalg.norm(plug_torque)

            # Update buffer
            buffer.record((prev_state, action, reward, obs))
            buffer.learn()

            update_target(actor_target.variables, actor_model.variables, tau)
            update_target(critic_target.variables, critic_model.variables, tau)

            # Logging
            episodic_reward += reward
            episodic_force += force_norm
            episodic_torque += torque_norm

            if step_count % logging_freq == 0:
                log_dict = {
                    'step': step_count,
                    'obs': obs,
                    'command': command,
                    'action': action,
                    'reward': reward,
                    'plug_force': plug_force,
                    'plug_torque': plug_torque,
                    'insertion_depth': insertion_depth,
                }

                for param, val in log_dict.items():
                    episode_log[param].append(val)

            step_count += 1

            # End this episode when `done` is True
            if done:
                break

            prev_state = obs

        # Log episode data
        episode_log = log_episode_data(neptune_run=run, episode_data=episode_log, episode=episode_count)

        ep_reward_list.append(episodic_reward)

        ep_avg_force = episodic_force / step_count
        ep_force_list.append(ep_avg_force)  # average force over this episode

        ep_avg_torque = episodic_torque / step_count
        ep_torque_list.append(ep_avg_torque)  # average torque over this episode

        avg_reward = np.mean(ep_reward_list[-100:])
        avg_reward_list.append(avg_reward)

        avg_force = np.mean(ep_force_list[-100:])
        avg_force_list.append(avg_force)

        avg_torque = np.mean(ep_torque_list[-100:])
        avg_torque_list.append(avg_torque)

        insert_depth = insertion_depth
        end_depth_list.append(insert_depth)
        avg_height = np.mean(end_depth_list[-100:])
        avg_height_list.append(avg_height)

        ep_logger = run['data']
        ep_data = {
            'ep_reward': episodic_reward,
            'ep_avg_force': ep_avg_force,
            'ep_avg_torque': ep_avg_torque,
            'ep_end_depth_x': insert_depth[0],
            'ep_end_depth_z': insert_depth[1],
            'ep_end_depth_rot': insert_depth[2],
        }

        for param, val in ep_data.items():
            ep_logger[param].append(val)

        # Print details about the performance of this episode
        print('\nReward')
        print(f'\tAvg per step: {episodic_reward / step_count}')
        print(f'\tTotal: {episodic_reward}')
        print(f'\tAvg from last 100 ep: {avg_reward}')  # Mean reward of last 100 episodes

        print('\nForces')
        print(f'\tAvg Force from last 100 ep: {avg_force}')  # Mean constraint force on plug of last 100 episodes
        print(f'\tAvg Torque from last 100 ep: {avg_torque}')  # Mean constraint force on plug of last 100 episodes

        print('\nInsertion depth')
        print(f'\tX: {insert_depth[0]}')
        print(f'\tZ: {insert_depth[1]}')
        print(f'\tRot: {insert_depth[2]}')
        print('')

    print('Last epoch reached.')
    exec_time = time.time() - training_start_time
    print(f'Time to execute: {exec_time} [{exec_time/total_episodes} avg.]')
