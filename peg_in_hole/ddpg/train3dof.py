import time
import logging
import tensorflow as tf
import gymnasium as gym
import numpy as np
import hydra
from omegaconf import DictConfig
import time


from peg_in_hole.settings import app_settings
from peg_in_hole.ddpg.buffer import OUActionNoise, Buffer, update_target
from peg_in_hole.ddpg.networks import get_actor, get_critic
import peg_in_hole.vortex_envs.vortex_interface  # noqa: F401 Needed to register env to gym
from peg_in_hole.utils.Neptune import NeptuneRun

# TODO: DEMAIN: Log at freq
# TODO: DEMAIN: Log ep data
# TODO: DEMAIN: Plots

logger = logging.getLogger(__name__)


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
    neptune_logger = NeptuneRun(neptune_cfg=cfg.neptune)
    neptune_logger.run['task_cfg'] = task_cfg

    # Create the env
    env_name = 'vxUnjamming-v0'
    render_mode = 'human' if cfg.render else None

    start_time = time.time()
    env = gym.make(env_name, render_mode=render_mode, neptune_logger=neptune_logger, task_cfg=task_cfg)
    print(f'init took: {time.time() - start_time} sec')

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
    ep_reward_list = []
    ep_force_list = []
    ep_torque_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    avg_force_list = []
    end_depth_list = []
    avg_torque_list = []
    avg_height_list = []

    start_time = time.time()

    # Training
    for ep in range(total_episodes):
        prev_state, _ = env.reset()
        # prev_state = prev_state[0]
        episodic_reward = 0
        episodic_force = 0
        episodic_torque = 0

        count = 0
        while True:
            if ep == total_episodes - 1:
                env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state, ou_noise, actor_model, lower_bound, upper_bound)

            # Recieve state and reward from environment.
            state, reward, done, truncated, info = env.step(action)

            force = env.unwrapped.get_plug_force()
            force_norm = np.sqrt(force.x**2.0 + force.y**2.0 + force.z**2.0)

            torque = env.unwrapped.get_plug_torque()
            torque_norm = np.sqrt(torque.x**2.0 + torque.y**2.0 + torque.z**2.0)

            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward
            episodic_force += force_norm
            episodic_torque += torque_norm

            buffer.learn()
            update_target(actor_target.variables, actor_model.variables, tau)
            update_target(critic_target.variables, critic_model.variables, tau)

            count += 1

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)
        ep_force_list.append(episodic_force / count)  # average force over this episode
        ep_torque_list.append(episodic_torque / count)  # average torque over this episode

        print('Avg reward per step over episode: ' + str(episodic_reward / count))
        print('Total reward over episode: ' + str(episodic_reward))

        # Mean reward of last 100 episodes
        avg_reward = np.mean(ep_reward_list[-100:])
        print('Episode * {} * Avg Reward from last 100 episodes is ==> {}'.format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

        # Mean constraint force on plug of last 100 episodes
        avg_force = np.mean(ep_force_list[-100:])
        print('Episode * {} * Avg Force is ==> {}'.format(ep, avg_force))
        avg_force_list.append(avg_force)

        # Mean constraint force on plug of last 100 episodes
        avg_torque = np.mean(ep_torque_list[-100:])
        print('Episode * {} * Avg Torque is ==> {}'.format(ep, avg_torque))
        avg_torque_list.append(avg_torque)

        # End depth for this episode
        insert_depth = env.unwrapped.get_insertion_depth()
        print('Episode * {} * Insertion Height is ==> {}'.format(ep, insert_depth))
        print('')
        end_depth_list.append(insert_depth)
        avg_height = np.mean(end_depth_list[-100:])
        avg_height_list.append(avg_height)

    print('Last epoch reached.')
    exec_time = time.time() - start_time
    print(f'Time to execute: {exec_time} [{exec_time/total_episodes} avg.]')
