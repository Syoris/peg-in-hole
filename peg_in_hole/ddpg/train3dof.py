import time
import logging
import tensorflow as tf
import gymnasium as gym
import numpy as np
import hydra

from peg_in_hole.settings import app_settings
from peg_in_hole.ddpg.buffer import OUActionNoise, Buffer, update_target
from peg_in_hole.ddpg.networks import get_actor, get_critic
import peg_in_hole.vortex_envs.vortex_interface  # noqa: F401 Needed to register env to gym

logger = logging.getLogger(__name__)

RENDER = True


def policy(state, noise_object, actor_model, lower_bound, upper_bound):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()

    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return np.squeeze(legal_action)


def train3dof():
    env_name = 'vxUnjamming-v0'
    if RENDER:
        render_mode = 'human'
    else:
        render_mode = None

    env = gym.make(env_name, render_mode=render_mode)

    num_states = env.observation_space.shape[0]
    print('Size of State Space ->  {}'.format(num_states))
    num_actions = env.action_space.shape[0]
    print('Size of Action Space ->  {}'.format(num_actions))

    upper_bound = env.action_space.high
    lower_bound = env.action_space.low

    print('Max Value of Action ->  {}'.format(upper_bound))
    print('Min Value of Action ->  {}'.format(lower_bound))

    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    actor_model = get_actor(num_states=num_states, num_actions=num_actions)
    critic_model = get_critic(num_states=num_states, num_actions=num_actions)

    actor_target = get_actor(num_states=num_states, num_actions=num_actions)
    critic_target = get_critic(num_states=num_states, num_actions=num_actions)

    # Making the weights equal initially
    actor_target.set_weights(actor_model.get_weights())
    critic_target.set_weights(critic_model.get_weights())

    # Learning rate for actor-critic models
    critic_lr = 0.0001
    actor_lr = 0.0001

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    total_episodes = 50
    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.001

    buffer_capacity = 200000
    batch_size = 32
    buffer = Buffer(
        num_states=num_states,
        num_actions=num_actions,
        actor_model=actor_model,
        target_actor=actor_target,
        actor_optimizer=actor_optimizer,
        critic_model=critic_model,
        target_critic=critic_target,
        critic_optimizer=critic_optimizer,
        gamma=gamma,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
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
