import numpy as np
import tensorflow as tf
import multiprocessing
from utils import *
import gym
import time
import copy
from random import randint

from osim.env import RunEnv

class Actor(multiprocessing.Process):
    def __init__(self, args, task_q, result_q, actor_id, monitor,
                 weights_ready_event):
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        self.args = args
        self.actor_id = actor_id
        self.monitor = monitor
        self.weights_ready_event = weights_ready_event

    def act(self, obs):
        obs = np.expand_dims(obs, 0)
        mean, logstd = self.session.run([self.action_mean, self.action_logstd],
                                        feed_dict={self.obs: obs})
        # samples the guassian distribution
        act = mean + np.exp(logstd) * np.random.randn(*logstd.shape)
        return act.ravel(), mean, logstd

    def run(self):
        if self.args.task == 'osim-rl':
            self.env = RunEnv(visualize=False)
        else:
            self.env = gym.make(self.args.task)
        self.env.seed(randint(0,999999))
        if self.monitor:
            self.env.monitor.start('monitor/', force=True)

        config = tf.ConfigProto(
            device_count={'GPU': 0},
            #gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1),
        )
        self.session = tf.Session(config=config)

        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = np.prod(self.env.action_space.shape)
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        batch_size = tf.shape(self.obs)[0]

        mean, logstd = policy_network(self.obs, self.observation_size,
                                      self.action_size, self.args.hidden_size,
                                      layer_norm=self.args.layer_norm,
                                      scope="actor%d" % self.actor_id)
        self.action_mean = mean
        logstd = tf.tile(logstd, (batch_size, 1))
        self.action_logstd = logstd

        self.session.run(tf.global_variables_initializer())
        var_list = [var for var in tf.trainable_variables()
                    if 'policy' in var.name]

        self.current_weights = None
        self.noise_vars = [tf.random_normal(shape=tf.shape(var),
                                            stddev=self.args.noise,
                                            seed=1234 + i)
                           for i, var in enumerate(var_list)]
        self.set_policy = SetPolicyWeights(self.session, var_list)

        while True:
            # get a task, or wait until it gets one
            next_task = self.task_q.get(block=True)
            if next_task == 1:
                # the task is an actor request to collect experience
                assert self.current_weights is not None
                path = self.rollout()
                self.task_q.task_done()
                self.result_q.put(path)
            elif next_task == 2:
                print "kill message"
                if self.monitor:
                    self.env.monitor.close()
                self.task_q.task_done()
                break
            else:
                # the task is to cache parameters of the actor policy
                self.current_weights = next_task
                noise = self.session.run(self.noise_vars)
                self.set_policy([v1 + v2 for v1, v2 in zip(self.current_weights,
                                                          noise)])
                self.task_q.task_done()
                self.weights_ready_event.wait()

    def rollout(self):
        obs, actions, rewards, action_means, action_logstds = [], [], [], [], []
        ob = self.env.reset()
        for i in xrange(self.args.max_pathlength - 1):
            obs.append(ob)
            action, action_mean, action_logstd = self.act(ob)
            actions.append(action)
            action_means.append(action_mean)
            action_logstds.append(action_logstd)
            [ob, reward, done, info]  = self.env.step(action)
            rewards.append((reward))
            if done or i == self.args.max_pathlength - 2:
                path = {
                    "obs": np.concatenate(np.expand_dims(obs, 0)),
                    "actions":  np.array(actions),
                    "action_mean": np.concatenate(action_means),
                    "action_logstd": np.concatenate(action_logstds),
                    "rewards": np.array(rewards),
                }
                return path

class ParallelRollout():
    def __init__(self, args):
        self.args = args

        self.tasks = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        self.weights_ready_event = multiprocessing.Event()

        self.actors = []
        self.actors.append(Actor(self.args, self.tasks, self.results, 0,
                                 args.monitor, self.weights_ready_event))

        for i in xrange(self.args.num_threads - 1):
            self.actors.append(Actor(self.args, self.tasks, self.results, i + 1,
                                     False, self.weights_ready_event))

        for a in self.actors:
            a.start()

        # initial estimate
        self.average_timesteps = 1000

    def rollout(self):
        num_rollouts = self.args.timesteps_per_batch / self.average_timesteps
        print("Number of rollouts: %d" % num_rollouts)

        for i in xrange(num_rollouts):
            self.tasks.put(1)
        self.tasks.join()

        paths = []
        while num_rollouts:
            num_rollouts -= 1
            paths.append(self.results.get())

        episode_rewards = np.array([path["rewards"].sum() for path in paths])
        print("%-40s: %f, %f" % ("Rewards (mean, std)", episode_rewards.mean(),
                                 episode_rewards.std()))

        episode_steps = np.array([len(path["rewards"]) for path in paths])
        print("%-40s: %d, %f" % ("Timesteps (total, average)",
                                 episode_steps.sum(), episode_steps.mean()))
        self.average_timesteps = int(episode_steps.mean())

        return paths

    def set_policy_weights(self, parameters):
        self.weights_ready_event.clear()
        for i in xrange(self.args.num_threads):
            self.tasks.put(parameters)
        self.tasks.join()
        self.weights_ready_event.set()

    def end(self):
        for i in xrange(self.args.num_threads):
            self.tasks.put(2)
