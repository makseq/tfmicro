#!/usr/bin/env python
import numpy as np
import tensorflow as tf


class AttentionWithContext(object):

    def __init__(self, shape, units, keep_prob=None, name='attention_context', verbose=False):
        with tf.variable_scope(name):
            init = tf.glorot_uniform_initializer()
            self.shape = shape
            self.name = name
            self.epsilon = 1e-10
            self.verbose = verbose
            self.units = units
            self.keep_prob = keep_prob
            dim = shape[-1]
            #self.W = tf.Variable(initial_value=init([dim, units]), dtype=tf.float32, name='W')
            #self.Wb = tf.Variable(initial_value=init([units]), dtype=tf.float32, name='b')
            self.U = tf.Variable(initial_value=init([units, dim]), dtype=tf.float32, name='U')
            self.Ub = tf.Variable(initial_value=init([dim]), dtype=tf.float32, name='b')
            self.rnn_cell = tf.nn.rnn_cell.GRUCell(units)

    def call(self, x):
        uit, _ = tf.nn.dynamic_rnn(self.rnn_cell, x, dtype=tf.float32)  # => [b*num_files, t, units[0]]
        if self.keep_prob is not None:
            uit = tf.nn.dropout(uit, keep_prob=self.keep_prob)

        # W
        # uit = tf.einsum('ijk,kl->ijl', hidden, self.W)  # [b, time, dim] x [dim, att] => [b, time, att]
        # uit += self.Wb  # => [b, time, att]
        # uit = tf.tanh(uit)  # => => [b, time, dim]
        # self.debug('uit shape', uit.get_shape())

        ait = tf.einsum('ijk,kl->ijl', uit, self.U)  # [b, time, att] x [att, dim] => [b, time, dim]
        ait += self.Ub  # => [b, time, att]
        self.debug('ait shape', ait.get_shape())

        a = tf.exp(ait)  # => [b, time, dim]
        a /= tf.reduce_sum(a, axis=1, keepdims=True) + self.epsilon  # => [b, time, dim]
        self.a = a

        weighted_input = x * a  # [b, time, dim] x [b, time, dim] => [b, time, dim]
        self.weighted_input = weighted_input
        self.debug('wighted input shape', weighted_input.get_shape())
        return tf.reduce_sum(weighted_input, axis=1)  # => [b, dim]

    def debug(self, *args):
        if self.debug:
            print self.name, '-->',
            for k in args:
                print k,
            print


def unit_test():
    sess = tf.Session()
    shape = [5, 3, 2]
    inp = np.arange(np.prod(shape)).reshape(shape)
    print '\n\ninput shape -->', inp.shape

    x = tf.placeholder(shape=shape, dtype=tf.float32)
    att = AttentionWithContext(shape, 256, verbose=True)
    out = att.call(x)

    sess.run(tf.global_variables_initializer())
    result = sess.run(out, feed_dict={x: inp})
    print 'output -->\n', result
    print 'output shape -->\n', result.shape


if __name__ == '__main__':
    unit_test()
