import opensim as osim
from osim.env import *

import numpy as np
import tensorflow as tf
import argparse

from utils import policy_network

# Command line parameters
parser = argparse.ArgumentParser(description='Submit the result to crowdAI')
parser.add_argument("--hidden_size", nargs="+", type=int, default=[64, 64])
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--visualize", action='store_true', default=False)
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()

env = RunEnv(visualize=args.visualize)

observation_size = env.observation_space.shape[0]
action_size = np.prod(env.action_space.shape)
obs_ph = tf.placeholder(tf.float32, [None, observation_size])
batch_size = tf.shape(obs_ph)[0]

scope = "policy"
action_mean, action_logstd = policy_network(obs_ph, observation_size,
                                            args.hidden_size, action_size,
                                            scope=scope)
action_logstd = tf.tile(action_logstd, (batch_size, 1))

saver = tf.train.Saver()
session = tf.Session()
saver.restore(session, args.model_path)

def act(obs):
    obs = np.expand_dims(obs, 0)
    mu, logstd = session.run([action_mean, action_logstd],
                             feed_dict={obs_ph: obs})
    # samples the guassian distribution
    act = mu + np.exp(logstd) * np.random.randn(*logstd.shape)
    return act.ravel()

episode_rewards = []
episode_steps = []
total = 0
steps = 0
resets = 0
observation = env.reset()
while True:
    action = act(observation)
    [observation, reward, done, info] = env.step(action.tolist())
    total += reward
    steps += 1
    if done:
        episode_rewards.append(total)
        episode_steps.append(steps)
        total = 0
        steps = 0
        resets += 1
        if resets == args.runs:
            break
        else:
            observation = env.reset()

print(episode_rewards)
print(episode_steps)
print("Mean: %f, Std: %f" % (np.mean(episode_rewards), np.std(episode_rewards)))
