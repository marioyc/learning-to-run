import tensorflow as tf
import numpy as np
from utils import *

class ValueFunction(object):
    def __init__(self, args, observation_size, session):
        self.args = args
        self.session = session

        hidden_size = 64
        self.x = tf.placeholder(tf.float32, shape=[None, observation_size], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")

        weight_init = normc_initializer(1.0)
        weight_regularizer = tf.contrib.layers.l2_regularizer(self.args.l2_reg)
        bias_init = tf.constant_initializer(0)

        with tf.variable_scope("VF"):
            h1 = fully_connected(self.x, observation_size, hidden_size,
                                 weight_init, weight_regularizer, bias_init, "h1")
            h1 = tf.nn.tanh(h1)
            h2 = fully_connected(h1, hidden_size, hidden_size, weight_init,
                                 weight_regularizer, bias_init, "h2")
            h2 = tf.nn.tanh(h2)
            h3 = fully_connected(h2, hidden_size, 1, weight_init,
                                 weight_regularizer, bias_init, "h3")
        self.vf = tf.reshape(h3, (-1,))

        vf_loss = tf.nn.l2_loss(self.vf - self.y)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="VF")

        self.global_step = tf.Variable(initial_value=0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.args.lr, self.global_step, 1, 0.9999)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.optimizer.minimize(vf_loss + sum(reg_losses))

    def fit(self, paths):
        featmat = np.concatenate([path["obs"] for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])

        indexes = np.arange(len(featmat))
        self.session.run(tf.assign(self.global_step, 0))
        for _ in range(self.args.epochs):
            start = 0
            np.random.shuffle(indexes)

            while start + self.args.batch_size < len(featmat):
                end = min(len(featmat), start + self.args.batch_size)
                minibatch_indexes = indexes[start:end]
                feed_dict = {
                    self.x: featmat[minibatch_indexes],
                    self.y: returns[minibatch_indexes]
                }
                self.session.run(self.train_op, feed_dict)
                start = end

    def predict(self, path):
        ret = self.session.run(self.vf, {self.x: path["obs"]})
        return np.reshape(ret, (ret.shape[0], ))
