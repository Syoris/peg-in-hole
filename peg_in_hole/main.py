from settings import app_settings
from peg_in_hole.vortex_envs.kinova_gen2_env import KinovaGen2Env
import time
import logging
import traceback
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np

"""
Time comp:
For 50 steps, w/ rendering
old code: 207.0949604511261 [4.141899209022522 avg.]
new code: 113.9709882736206 [2.279419765472412 avg.]
new version: 
"""

logger = logging.getLogger(__name__)

env_name = 'vxUnjamming-v0'
env = gym.make(env_name)
# kinova_env = KinovaGen2Env()

num_states = env.observation_space.shape[0]
print('Size of State Space ->  {}'.format(num_states))
num_actions = env.action_space.shape[0]
print('Size of Action Space ->  {}'.format(num_actions))

upper_bound = env.action_space.high
lower_bound = env.action_space.low

print('Max Value of Action ->  {}'.format(upper_bound))
print('Min Value of Action ->  {}'.format(lower_bound))


def get_actor():
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(32, activation='relu')(inputs)
    out = layers.Dense(32, activation='relu')(out)
    out = layers.Dense(32, activation='relu')(out)
    out = layers.Dense(32, activation='relu')(out)
    outputs = layers.Dense(num_actions, activation='tanh')(out)

    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    # Action as input
    action_input = layers.Input(shape=(num_actions))

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_input, action_input])

    out = layers.Dense(32, activation='relu')(concat)
    out = layers.Dense(32, activation='relu')(out)
    out = layers.Dense(32, activation='relu')(out)
    out = layers.Dense(32, activation='relu')(out)

    outputs = layers.Dense(1, activation='linear')(out)

    # Outputs single value for given state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()

    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return np.squeeze(legal_action)


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=1000000, batch_size=32):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1][0]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions], training=True)
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


@tf.function
def update_target(target_weights, weights, tau):
    for a, b in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

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

buffer = Buffer(200000, 32)


def train_ddpg():
    "test function"

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
            # Uncomment this to see the Actor in action
            # But not in a python notebook.

            if ep == total_episodes - 1:
                env.render()

            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

            action = policy(tf_prev_state, ou_noise)

            # Recieve state and reward from environment.
            state, reward, done, truncated, info = env.step(action)

            force = env.get_plug_force()
            force_norm = np.sqrt(force.x**2.0 + force.y**2.0 + force.z**2.0)

            torque = env.get_plug_torque()
            torque_norm = np.sqrt(torque.x**2.0 + torque.y**2.0 + torque.z**2.0)

            buffer.record((prev_state, action, reward, state))
            episodic_reward += reward
            episodic_force += force_norm
            episodic_torque += torque_norm

            buffer.learn()
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

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
        insert_depth = env.get_insertion_depth()
        print('Episode * {} * Insertion Height is ==> {}'.format(ep, insert_depth))
        print('')
        end_depth_list.append(insert_depth)
        avg_height = np.mean(end_depth_list[-100:])
        avg_height_list.append(avg_height)

    print('Last epoch reached.')
    exec_time = time.time() - start_time
    print(f'Time to execute: {exec_time} [{exec_time/total_episodes} avg.]')


if __name__ == '__main__':
    logger.info('---------------- Peg-in-hole Package ----------------')
    try:
        train_ddpg()
    except RuntimeError as e:
        logger.error(e, exc_info=True)
        raise e

    except Exception as e:  # noqa
        logger.error('uncaught exception: %s', traceback.format_exc())
        raise e

    logger.info('Done')
