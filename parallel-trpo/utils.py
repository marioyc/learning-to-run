import tensorflow as tf
import numpy as np
import scipy.signal

# KL divergence with itself, holding first argument fixed
def gauss_selfKL_firstfixed(mu, logstd):
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd

    return gauss_KL(mu1, logstd1, mu2, logstd2)

# probability to take action x, given paramaterized guassian distribution
def gauss_log_prob(mu, logstd, x):
    var = tf.exp(2*logstd)
    gp = -tf.square(x - mu)/(2*var) - .5*tf.log(tf.constant(2*np.pi)) - logstd
    return  tf.reduce_sum(gp, [1])

# KL divergence between two paramaterized guassian distributions
def gauss_KL(mu1, logstd1, mu2, logstd2):
    var1 = tf.exp(2*logstd1)
    var2 = tf.exp(2*logstd2)

    kl = tf.reduce_sum(logstd2 - logstd1 + (var1 + tf.square(mu1 - mu2))/(2*var2) - 0.5)
    return kl

# Shannon entropy for a paramaterized guassian distributions
def gauss_ent(mu, logstd):
    h = tf.reduce_sum(logstd + tf.constant(0.5*np.log(2*np.pi*np.e), tf.float32))
    return h

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

def xavier_initializer(self, shape):
    dim_sum = np.sum(shape)
    if len(shape) == 1:
        dim_sum += 1
    bound = np.sqrt(6.0 / dim_sum)
    return tf.random_uniform(shape, minval=-bound, maxval=bound)

def fully_connected(input_layer, input_size, output_size,
                    weight_init, weight_regularizer, bias_init, scope):
    with tf.variable_scope(scope):
        w = tf.get_variable("w", [input_size, output_size],
                            initializer=weight_init,
                            regularizer=weight_regularizer)
        b = tf.get_variable("b", [output_size], initializer=bias_init)
    return tf.matmul(input_layer,w) + b
