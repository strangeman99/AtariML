from __future__ import absolute_import, division, print_function
import gym
import base64
import imageio
import IPython
import numpy as np
import PIL.Image
import pyvirtualdisplay

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tensorflow import keras
from keras import layers


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
#def dense_layer(num_units):
    #return tf.keras.layers.Dense(
        #num_units,
        #activation=tf.keras.activations.relu,
        #kernel_initializer=tf.keras.initializers.VarianceScaling(
            #scale=2.0, mode='fan_in', distribution='truncated_normal'))

def compute_avg_return(environment, policy, num_episodes):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def main():
    learning_rate = 1e-3
    tf.version.VERSION

    env_name = "Breakout-v4"
    env = suite_gym.load(env_name)

    print('Observation Spec:')  # just print things out so I know the environment
    print(env.time_step_spec().observation)
    print('Reward Spec:')
    print(env.time_step_spec().reward)
    print('Action Spec:')
    print(env.action_spec())
    time_step = env.reset()
    print('Time step:')
    print(time_step)

    action = np.array(3, dtype=np.int32)

    next_time_step = env.step(action)
    print('Next time step:')
    print(next_time_step)

    train_py_env = suite_gym.load(env_name)
    eval_py_env = suite_gym.load(env_name)
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    fc_layer_params = (10, 5)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
    print(num_actions)

    def create_q_model():
        inputs = layers.Input(shape=(84, 84, 4))

        layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = layers.Flatten()(layer3)

        layer5 = layers.Dense(512, activation="relu")(layer4)
        theAction = layers.Dense(num_actions, activation="linear")(layer5)

        return keras.Model(inputs=inputs, outputs=theAction)

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output. This wasn't working for me. I'm making my own with Keras
    #dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    #q_values_layer = tf.keras.layers.Dense(num_actions, activation=None, kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03), bias_initializer=tf.keras.initializers.Constant(-0.2))
    #q_net = sequential.Sequential(dense_layers + [q_values_layer])

    q_net = create_q_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy =agent.collect_policy

    # testing the random policy and seeing how it works
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec());
    example_environment = tf_py_environment.TFPyEnvironment(suite_gym.load(env_name))
    time_step = example_environment.reset()
    random_policy.action(time_step)
    num_eval_episodes = 1000
    compute_avg_return(eval_policy, random_policy, num_eval_episodes)


    env.close()

if __name__ == '__main__':
    main()

