from keras import layers
import tensorflow as tf


def get_actor(num_states: int, num_actions: int):
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(32, activation='relu')(inputs)
    out = layers.Dense(32, activation='relu')(out)
    out = layers.Dense(32, activation='relu')(out)
    out = layers.Dense(32, activation='relu')(out)
    outputs = layers.Dense(num_actions, activation='tanh')(out)

    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(num_states: int, num_actions: int):
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
