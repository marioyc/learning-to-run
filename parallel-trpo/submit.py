import opensim as osim
from osim.http.client import Client
from osim.env import *

import numpy as np
import tensorflow as tf
import argparse

from utils import fully_connected

# Settings
remote_base = 'http://grader.crowdai.org:1729'

# Command line parameters
parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument('--token', dest='token', action='store', required=True)
parser.add_argument("--visualize", type=bool, default=False)
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

env = RunEnv(visualize=args.visualize)
client = Client(remote_base)

observation_size = env.observation_space.shape[0]
action_size = np.prod(env.action_space.shape)
hidden_size = 64
obs_placeholder = tf.placeholder(tf.float32, [None, observation_size])

weight_init = tf.random_uniform_initializer(-0.05, 0.05)
weight_regularizer = tf.contrib.layers.l2_regularizer(0.0)
bias_init = tf.constant_initializer(0)

# Load model
with tf.variable_scope("policy"):
    h1 = fully_connected(obs_placeholder, observation_size, hidden_size,
                         weight_init, weight_regularizer, bias_init, "policy_h1")
    h1 = tf.nn.relu(h1)
    h2 = fully_connected(h1, hidden_size, hidden_size,
                         weight_init, weight_regularizer, bias_init, "policy_h2")
    h2 = tf.nn.relu(h2)
    h3 = fully_connected(h2, hidden_size, action_size,
                         weight_init, weight_regularizer, bias_init, "policy_h3")
    action_dist_logstd_param = tf.Variable((.01*np.random.randn(1, action_size)).astype(np.float32), name="policy_logstd")

action_dist_mu = h3
action_dist_logstd = tf.tile(action_dist_logstd_param, tf.stack((tf.shape(action_dist_mu)[0], 1)))

saver = tf.train.Saver()
session = tf.Session()
saver.restore(session, args.model_path)

def act(obs):
    obs = np.expand_dims(obs, 0)
    mu, logstd = session.run([action_dist_mu, action_dist_logstd], feed_dict={obs_placeholder: obs})
    # samples the guassian distribution
    act = mu + np.exp(logstd) * np.random.randn(*logstd.shape)
    return act.ravel()

episode_rewards = []
episode_steps = []
total = 0
steps = 0
resets = 0
# Create environment
observation = client.env_create(args.token)

while True:
    action = act(observation)
    [observation, reward, done, info] = client.env_step(action.tolist())
    total += reward
    steps += 1
    if done:
        print("RESET")
        episode_rewards.append(total)
        episode_steps.append(steps)
        total = 0
        steps = 0
        observation = client.env_reset()
        if not observation:
            break

client.submit()

print(episode_rewards)
print(episode_steps)
print("Mean: %f, Std: %f" % (np.mean(episode_rewards), np.std(episode_rewards)))
