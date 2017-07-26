import numpy as np
import tensorflow as tf
import gym
from utils import *
from model import *
import argparse
from rollouts import *
import json

from osim.env import RunEnv

parser = argparse.ArgumentParser(description='PPO.')
# these parameters should stay the same
parser.add_argument("--task", type=str, default='osim-rl')
parser.add_argument("--timesteps_per_batch", type=int, default=5000)
parser.add_argument("--n_steps", type=int, default=1000000)
parser.add_argument("--gamma", type=float, default=.99)
parser.add_argument("--num_threads", type=int, default=8)
parser.add_argument("--monitor", type=bool, default=False)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epsilon", type=float, default=0.2)
parser.add_argument("--log_name", type=str, default='ppo')
parser.add_argument("--l2_reg", type=float, default=0.0)
parser.add_argument("--save_steps", type=int, default=100)

args = parser.parse_args()

if args.task == 'osim-rl':
    args.max_pathlength = 1000
else:
    args.max_pathlength = gym.spec(args.task).timestep_limit

learner_tasks = multiprocessing.JoinableQueue()
learner_results = multiprocessing.Queue()

if args.task == 'osim-rl':
    learner_env = RunEnv(visualize=False)
else:
    learner_env = gym.make(args.task)

learner = PPO(args, learner_env.observation_space, learner_env.action_space, learner_tasks, learner_results)
learner.start()
rollouts = ParallelRollout(args)

learner_tasks.put(1)
learner_tasks.join()
starting_weights = learner_results.get()
rollouts.set_policy_weights(starting_weights)

history = {}
history["rollout_time"] = []
history["learn_time"] = []
history["mean_reward"] = []
history["timesteps"] = []

start_time = time.time()
totalsteps = 0
iteration = 0
while True:
    iteration += 1
    print "-------- Iteration %d ----------" % iteration

    # runs a bunch of async processes that collect rollouts
    rollout_start = time.time()
    paths = rollouts.rollout()
    rollout_time = (time.time() - rollout_start) / 60.0

    # Why is the learner in an async process?
    # Well, it turns out tensorflow has an issue: when there's a tf.Session in the main thread
    # and an async process creates another tf.Session, it will freeze up.
    # To solve this, we just make the learner's tf.Session in its own async process,
    # and wait until the learner's done before continuing the main thread.
    learn_start = time.time()
    learner_tasks.put(paths)
    learner_tasks.join()
    new_policy_weights, mean_reward = learner_results.get()
    learn_time = (time.time() - learn_start) / 60.0
    print "Total time: %.2f mins" % ((time.time() - start_time) / 60.0)

    history["rollout_time"].append(rollout_time)
    history["learn_time"].append(learn_time)
    history["mean_reward"].append(mean_reward)
    history["timesteps"].append(args.timesteps_per_batch)

    print "Current steps is " + str(args.timesteps_per_batch)

    if iteration % args.save_steps == 0:
        with open("%s" % args.task, "w") as outfile:
            json.dump(history, outfile)
        learner_tasks.put(2)

    totalsteps += args.timesteps_per_batch
    print "%d total steps have happened\n" % totalsteps
    if totalsteps > args.n_steps:
        break

    rollouts.set_policy_weights(new_policy_weights)

rollouts.end()
learner_tasks.put(2)
learner_tasks.put(None)
