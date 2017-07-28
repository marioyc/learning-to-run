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
        self.observation_size = self.observation_space.shape[0]
        self.action_size = np.prod(self.action_space.shape)
        self.hidden_size = 64

        weight_init = tf.random_uniform_initializer(-0.05, 0.05)
        weight_regularizer = tf.contrib.layers.l2_regularizer(self.args.l2_reg)
        bias_init = tf.constant_initializer(0)

        config = tf.ConfigProto(
            device_count = {'GPU': 0}
        )
        self.session = tf.Session(config=config)

        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.advantage = tf.placeholder(tf.float32, [None])
        self.oldaction_dist_mu = tf.placeholder(tf.float32, [None, self.action_size])
        self.oldaction_dist_logstd = tf.placeholder(tf.float32, [None, self.action_size])

        with tf.variable_scope("policy"):
            h1 = fully_connected(self.obs, self.observation_size, self.hidden_size,
                                 weight_init, weight_regularizer, bias_init, "policy_h1")
            h1 = tf.nn.relu(h1)
            h2 = fully_connected(h1, self.hidden_size, self.hidden_size,
                                 weight_init, weight_regularizer, bias_init, "policy_h2")
            h2 = tf.nn.relu(h2)
            h3 = fully_connected(h2, self.hidden_size, self.action_size,
                                 weight_init, weight_regularizer, bias_init, "policy_h3")
            action_dist_logstd_param = tf.Variable((.01*np.random.randn(1, self.action_size)).astype(np.float32), name="policy_logstd")

        # means for each action
        self.action_dist_mu = h3
        # log standard deviations for each actions
        self.action_dist_logstd = tf.tile(action_dist_logstd_param, tf.stack((tf.shape(self.action_dist_mu)[0], 1)))

        batch_size = tf.shape(self.obs)[0]
        # what are the probabilities of taking self.action, given new and old distributions
        log_p_n = gauss_log_prob(self.action_dist_mu, self.action_dist_logstd, self.action)
        log_oldp_n = gauss_log_prob(self.oldaction_dist_mu, self.oldaction_dist_logstd, self.action)

        # tf.exp(log_p_n) / tf.exp(log_oldp_n)
        ratio = tf.exp(log_p_n - log_oldp_n)

        surr1 = ratio * self.advantage
        surr2 = tf.clip_by_value(ratio, 1.0 - self.args.epsilon, 1.0 + self.args.epsilon) * self.advantage
        pol_surr = -tf.reduce_mean(tf.minimum(surr1, surr2))
        var_list = tf.trainable_variables()

        batch_size_float = tf.cast(batch_size, tf.float32)
        # kl divergence and shannon entropy
        kl = gauss_KL(self.oldaction_dist_mu, self.oldaction_dist_logstd, self.action_dist_mu, self.action_dist_logstd) / batch_size_float
        ent = gauss_ent(self.action_dist_mu, self.action_dist_logstd) / batch_size_float

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="policy")

        self.losses = [pol_surr, kl, ent]

        self.global_step = tf.Variable(initial_value=0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.args.lr, self.global_step, 1, 0.9999)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.optimizer.minimize(pol_surr + sum(reg_losses))

        # value function
        self.vf = ValueFunction(self.args, self.observation_size, self.session)

        self.session.run(tf.global_variables_initializer())

        self.get_policy = GetPolicyWeights(self.session, var_list)

        # Logs
        self.saver = tf.train.Saver()
        self.save_count = 0

        self.writer = tf.summary.FileWriter('logs/' + self.args.log_name, self.session.graph)
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

        # is it possible to replace A(s,a) with Q(s,a)?
        for path in paths:
            path["baseline"] = self.vf.predict(path)
            path["returns"] = discount(path["rewards"], self.args.gamma)
            path["advantage"] = path["returns"] - path["baseline"]
            path["gae"] = calc_gae(path["rewards"], path["baseline"],
                                   self.args.gamma, self.args.lamb)
            # path["advantage"] = path["returns"]

        # puts all the experiences in a matrix: total_timesteps x options
        action_dist_mu = np.concatenate([path["action_dists_mu"] for path in paths])
        action_dist_logstd = np.concatenate([path["action_dists_logstd"] for path in paths])
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
            while start + self.args.batch_size < len(obs_n):
                end = min(len(obs_n), start + self.args.batch_size)
                minibatch_indexes = indexes[start:end]

                feed_dict = {
                    self.obs: obs_n[minibatch_indexes],
                    self.action: action_n[minibatch_indexes],
                    self.advantage: advant_n[minibatch_indexes],
                    self.oldaction_dist_mu: action_dist_mu[minibatch_indexes],
                    self.oldaction_dist_logstd: action_dist_logstd[minibatch_indexes],
                }
                _, minibatch_loss = self.session.run([self.train_op, self.losses[0]], feed_dict)

                mean_loss += minibatch_loss
                n_minibatches += 1
                updates += 1
                start = end

            mean_loss /= n_minibatches
            #summary_loss = self.session.run(self.summary_loss, feed_dict={self.mean_loss: mean_loss})
            #self.writer.add_summary(summary_loss, self.log_step_loss)
            #self.log_step_loss += 1

        print("Updates: %d" % updates)
        feed_dict = {
            self.obs: obs_n,
            self.action: action_n,
            self.advantage: advant_n,
            self.oldaction_dist_mu: action_dist_mu,
            self.oldaction_dist_logstd: action_dist_logstd,
        }
        surrogate_after, kl_after, entropy_after = self.session.run(self.losses, feed_dict)

        episoderewards = np.array([path["rewards"].sum() for path in paths])
        mean_reward = episoderewards.mean()
        summary_reward = self.session.run(self.summary_reward, feed_dict={self.mean_reward: mean_reward})
        self.writer.add_summary(summary_reward, self.log_step_reward)
        self.log_step_reward += 1

        stats = {}
        stats["Average sum of rewards"] = mean_reward
        stats["Std of sum of rewards"] = np.std(episoderewards)
        stats["Entropy"] = entropy_after
        stats["Timesteps"] = sum([len(path["rewards"]) for path in paths])
        stats["KL between old and new distribution"] = kl_after
        stats["Surrogate loss"] = surrogate_after
        for k, v in stats.iteritems():
            print(k + ": " + " " * (40 - len(k)) + str(v))

        return mean_reward
