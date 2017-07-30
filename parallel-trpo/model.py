import numpy as np
import tensorflow as tf
import gym
from utils import *
from rollouts import *
from value_function import *
import time
import os
import logging
import random
import multiprocessing

class PPO(multiprocessing.Process):
    def __init__(self, args, observation_space, action_space, task_q, result_q):
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        self.observation_space = observation_space
        self.action_space = action_space
        self.args = args

    def makeModel(self):
        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        self.session = tf.Session(config=config)

        self.observation_size = self.observation_space.shape[0]
        self.action_size = np.prod(self.action_space.shape)
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.advantage = tf.placeholder(tf.float32, [None])
        self.oldaction_mean = tf.placeholder(tf.float32, [None, self.action_size])
        self.oldaction_logstd = tf.placeholder(tf.float32, [None, self.action_size])
        batch_size = tf.shape(self.obs)[0]

        scope = "policy"
        mean, logstd = policy_network(self.obs, self.observation_size,
                                      self.action_size, l2_reg=self.args.l2_reg,
                                      scope=scope)
        self.action_mean = mean
        self.action_logstd = logstd
        logstd = tf.tile(logstd, (batch_size, 1))

        # what are the probabilities of taking self.action, given new and old distributions
        log_p = gauss_log_prob(mean, logstd, self.action)
        log_oldp = gauss_log_prob(self.oldaction_mean, self.oldaction_logstd,
                                  self.action)
        ratio = tf.exp(log_p - log_oldp)

        surr1 = ratio * self.advantage
        surr2 = tf.clip_by_value(ratio, 1.0 - self.args.epsilon,
                                 1.0 + self.args.epsilon) * self.advantage
        pol_surr = -tf.reduce_mean(tf.minimum(surr1, surr2))
        var_list = tf.trainable_variables()

        # kl divergence and shannon entropy
        kl = gauss_KL(self.oldaction_mean, self.oldaction_logstd, mean, logstd)
        ent = gauss_ent(mean, logstd)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                       scope=scope)
        reg_losses = sum(reg_losses)
        self.losses = [pol_surr, kl, ent]

        self.global_step = tf.Variable(initial_value=0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.args.lr,
                                                   self.global_step, 1, 0.9999)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.optimizer.minimize(pol_surr + reg_losses)

        # value function
        self.vf = ValueFunction(self.args, self.observation_size, self.session)

        self.session.run(tf.global_variables_initializer())

        self.get_policy = GetPolicyWeights(self.session, var_list)

        # Logs
        self.saver = tf.train.Saver()
        self.save_count = 0

        self.writer = tf.summary.FileWriter('logs/' + self.args.log_name,
                                            self.session.graph)
        self.mean_loss = tf.placeholder(tf.float32, shape=())
        self.summary_loss = tf.summary.scalar('mean_loss', self.mean_loss)
        self.log_step_loss = 0
        self.mean_reward = tf.placeholder(tf.float32, shape=())
        self.summary_reward = tf.summary.scalar('mean_reward', self.mean_reward)
        self.log_step_reward = 0

    def run(self):
        self.makeModel()
        while True:
            paths = self.task_q.get()
            if paths is None:
                # kill the learner
                self.task_q.task_done()
                break
            elif paths == 1:
                # just get params, no learn
                self.task_q.task_done()
                self.result_q.put(self.get_policy())
            elif paths == 2:
                # Save the model
                path = "models/%s-%d.ckpt" % (self.args.log_name, self.save_count)
                save_path = self.saver.save(self.session, path)
                print("Model saved in file: %s" % save_path)
                self.save_count += 1
                self.task_q.task_done()
            else:
                mean_reward = self.learn(paths)
                self.task_q.task_done()
                self.result_q.put((self.get_policy(), mean_reward))
        return

    def learn(self, paths):
        for path in paths:
            path["baseline"] = self.vf.predict(path)
            path["returns"] = discount(path["rewards"], self.args.gamma)
            path["advantage"] = path["returns"] - path["baseline"]
            path["gae"] = calc_gae(path["rewards"], path["baseline"],
                                   self.args.gamma, self.args.lamb)

        # puts all the experiences in a matrix: total_timesteps x options
        action_mean = np.concatenate([path["action_mean"] for path in paths])
        action_logstd = np.concatenate([path["action_logstd"] for path in paths])
        obs_n = np.concatenate([path["obs"] for path in paths])
        action_n = np.concatenate([path["actions"] for path in paths])

        # standardize to mean 0 stddev 1
        advant_n = np.concatenate([path["gae"] for path in paths])
        advant_n = (advant_n - advant_n.mean()) / advant_n.std()

        # train value function / baseline on rollout paths
        self.vf.fit(paths)

        indexes = np.arange(len(obs_n))
        self.session.run(tf.assign(self.global_step, 0))
        updates = 0
        for i in range(self.args.epochs):
            mean_loss = 0.0
            start = 0
            n_minibatches = 0
            np.random.shuffle(indexes)
            while start + self.args.batch_size <= len(obs_n):
                end = start + self.args.batch_size
                minibatch_indexes = indexes[start:end]

                feed_dict = {
                    self.obs: obs_n[minibatch_indexes],
                    self.action: action_n[minibatch_indexes],
                    self.oldaction_mean: action_mean[minibatch_indexes],
                    self.oldaction_logstd: action_logstd[minibatch_indexes],
                    self.advantage: advant_n[minibatch_indexes],
                }
                _, minibatch_loss = self.session.run([self.train_op,
                                                      self.losses[0]], feed_dict)

                mean_loss += minibatch_loss
                n_minibatches += 1
                updates += 1
                start = end

            mean_loss /= n_minibatches
            #summary_loss = self.session.run(self.summary_loss,
            #                                feed_dict={self.mean_loss: mean_loss})
            #self.writer.add_summary(summary_loss, self.log_step_loss)
            #self.log_step_loss += 1

        feed_dict = {
            self.obs: obs_n,
            self.action: action_n,
            self.oldaction_mean: action_mean,
            self.oldaction_logstd: action_logstd,
            self.advantage: advant_n,
        }
        surrogate_after, kl_after, entropy_after = self.session.run(self.losses,
                                                                    feed_dict)
        logstd = self.session.run(self.action_logstd)

        episoderewards = np.array([path["rewards"].sum() for path in paths])
        mean_reward = episoderewards.mean()
        summary_reward = self.session.run(self.summary_reward,
                                          feed_dict={self.mean_reward: mean_reward})
        self.writer.add_summary(summary_reward, self.log_step_reward)
        self.log_step_reward += 1

        timesteps = [len(path["rewards"]) for path in paths]

        stats = {}
        stats["Rewards (mean, std)"] = (mean_reward, np.std(episoderewards))
        stats["Timesteps (total, average)"] = (sum(timesteps),
                                               float(sum(timesteps)) / len(paths))
        stats["Updates"] = updates
        stats["KL between old and new distribution"] = kl_after
        stats["Action distribution (entropy, std)"] = (entropy_after,
                                                       np.exp(logstd[0]).tolist())
        for k, v in stats.iteritems():
            print(k + ": " + " " * (40 - len(k)) + str(v))

        return mean_reward
