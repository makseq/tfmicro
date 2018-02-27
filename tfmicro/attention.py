#!/usr/bin/env python
import numpy as np
import tensorflow as tf


class AttentionWithContext(object):

    def __init__(self, shape, units, name='attention_context', verbose=False):
        with tf.name_scope(name):
            init = tf.glorot_uniform_initializer()
            self.shape = shape
            self.name = name
            self.epsilon = 1e-10
            self.verbose = verbose
            self.units = units

            self.W = tf.Variable(initial_value=init([shape[-1], units]), dtype=tf.float32, name='W')
            self.b = tf.Variable(initial_value=init([units]), dtype=tf.float32, name='b')
            self.u = tf.Variable(initial_value=init([units, 1]), dtype=tf.float32, name='U')

    def call(self, x):
        uit = tf.einsum('ijk,kl->ijl', x, self.W)  # [b, time, dim] x [dim, dim] => [b, time, dim]
        uit += self.b  # => [b, time, dim]
        self.debug('uit shape', uit.get_shape())

        uit = tf.tanh(uit)  # => => [b, time, dim]
        ait = tf.einsum('ijk,kl->ijl', uit, self.u)  # [b, time, dim] x [dim, 1] => [b, time, 1]
        self.debug('ait shape', ait.get_shape())

        a = tf.exp(ait)  # => [b, time, 1]
        a /= tf.reduce_sum(a, axis=1, keep_dims=True) + self.epsilon  # => [b, time, 1]

        weighted_input = x * a  # [b, time, dim] x [b, time, 1] => [b, time, dim]
        self.debug('wighted input shape', weighted_input.get_shape())
        return tf.reduce_sum(weighted_input, axis=1)  # => [b, dim]'''

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
