import tensorflow as tf
import numpy as np
import scipy.signal

# probability to take action x, given paramaterized guassian distribution
def gauss_log_prob(mu, logstd, x):
    var = tf.exp(2 * logstd)
    gp = -tf.square(x - mu) / (2 * var) - 0.5 * np.log(2 * np.pi) - logstd
    return  tf.reduce_sum(gp, 1)

# KL divergence between two paramaterized guassian distributions
def gauss_KL(mu1, logstd1, mu2, logstd2):
    var1 = tf.exp(2 * logstd1)
    var2 = tf.exp(2 * logstd2)
    kl = tf.reduce_sum(logstd2 - logstd1 +
                       (var1 + tf.square(mu1 - mu2)) / (2 * var2) - 0.5, 1)
    return tf.reduce_mean(kl)

# Shannon entropy for a paramaterized guassian distributions
def gauss_ent(mu, logstd):
    h = tf.reduce_sum(logstd + 0.5 * np.log(2 * np.pi * np.e), 1)
    return tf.reduce_mean(h)

def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def calc_gae(rewards, value, gamma, lamb):
    T = len(rewards)
    gae = np.empty(T)
    gae[T - 1] = rewards[T - 1]
    for t in reversed(range(T - 1)):
        delta = rewards[t] + gamma * value[t + 1] - value[t]
        gae[t] = delta + gamma * lamb * gae[t + 1]
    return gae

class GetPolicyWeights(object):
    def __init__(self, session, var_list):
        self.session = session
        self.op = [var for var in var_list if 'policy' in var.name]
    def __call__(self):
        return self.session.run(self.op)

class SetPolicyWeights(object):
    def __init__(self, session, var_list):
        self.session = session
        self.policy_vars = [var for var in var_list if 'policy' in var.name]
        self.placeholders = {}
        self.assigns = []
        for var in self.policy_vars:
            self.placeholders[var.name] = tf.placeholder(tf.float32, var.get_shape())
            self.assigns.append(tf.assign(var,self.placeholders[var.name]))
    def __call__(self, weights):
        feed_dict = {}
        count = 0
        for var in self.policy_vars:
            feed_dict[self.placeholders[var.name]] = weights[count]
            count += 1
        self.session.run(self.assigns, feed_dict)

def fully_connected(input_layer, input_size, output_size,
                    weight_init, weight_regularizer, bias_init, scope):
    with tf.variable_scope(scope):
        w = tf.get_variable("w", [input_size, output_size],
                            initializer=weight_init,
                            regularizer=weight_regularizer)
        b = tf.get_variable("b", [output_size], initializer=bias_init)
    return tf.matmul(input_layer,w) + b

def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def policy_network(input_layer, input_size, action_size, mean_hidden_size=64,
                   l2_reg=0.0, scope="policy"):
    with tf.variable_scope(scope):
        weight_init = normc_initializer(1.0)
        weight_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        bias_init = tf.constant_initializer(0)

        # Mean
        mean_h = fully_connected(input_layer, input_size, mean_hidden_size,
                                 weight_init, weight_regularizer, bias_init,
                                 "policy_mean_h1")
        mean_h = tf.nn.tanh(mean_h)
        mean_h = fully_connected(mean_h, mean_hidden_size, mean_hidden_size,
                                 weight_init, weight_regularizer, bias_init,
                                 "policy_mean_h2")
        mean_h = tf.nn.tanh(mean_h)
        mean_h = fully_connected(mean_h, mean_hidden_size, action_size,
                                 normc_initializer(0.01), weight_regularizer,
                                 bias_init, "policy_mean_h3")

        # Std
        logstd = tf.Variable(tf.zeros([1, action_size]), name="policy_logstd")

    return mean_h, logstd
