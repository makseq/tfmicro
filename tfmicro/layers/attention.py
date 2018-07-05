#!/usr/bin/env python
"""
TFMicro
Copyright (C) 2018 Maxim Tkachenko

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import tensorflow as tf


class AttentionWithContext(object):

    def __init__(self, shape, units, keep_prob=None, name='attention_context', use_rnn=True, swap_memory=False, verbose=False):
        with tf.variable_scope(name):
            init = tf.glorot_uniform_initializer()
            self.shape = shape
            self.name = name
            self.epsilon = 1e-10
            self.verbose = verbose
            self.units = units
            self.keep_prob = keep_prob
            dim = int(shape[-1]) if isinstance(shape[-1], tf.Dimension) else shape[-1]

            self.use_rnn = use_rnn
            if use_rnn:
                self.rnn_cell = tf.nn.rnn_cell.GRUCell(units)
            else:
                self.W = tf.Variable(initial_value=init([dim, units]), dtype=tf.float32, name='W')
                self.Wb = tf.Variable(initial_value=init([units]), dtype=tf.float32, name='b')

            self.U = tf.Variable(initial_value=init([units, dim]), dtype=tf.float32, name='U')
            self.Ub = tf.Variable(initial_value=init([dim]), dtype=tf.float32, name='b')
            self.swap_memory = swap_memory

    def __call__(self, x):
        with tf.name_scope(self.name):
            # RNN attention
            if self.use_rnn:
                uit, _ = tf.nn.dynamic_rnn(self.rnn_cell, x, dtype=tf.float32, swap_memory=self.swap_memory)  # => [b*num_files, t, units[0]]
                if self.keep_prob is not None:
                    uit = tf.nn.dropout(uit, keep_prob=self.keep_prob)
                    self.debug('dropout enabled')

            # W simple attention
            else:
                uit = tf.einsum('ijk,kl->ijl', x, self.W)  # [b, time, dim] x [dim, att] => [b, time, att]
                uit += self.Wb  # => [b, time, att]
                uit = tf.tanh(uit)  # => => [b, time, dim]
                self.debug('uit shape', uit.get_shape())

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

    def call(self, x):
        return self.__call__(x)

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
