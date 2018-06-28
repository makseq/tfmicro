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


class TimeDistributed:
    def __init__(self, input_shape, output_dim, name, use_biases=True):
        self.name = name
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.use_biases = use_biases
        self.layer = None

        init = tf.glorot_uniform_initializer()
        with tf.variable_scope(name):
            self.weights = tf.Variable(init([int(input_shape[-1]), output_dim]), dtype=tf.float32)
            self.biases = tf.Variable(init([output_dim]))

    def __call__(self, x):
        with tf.variable_scope(self.name):
            # [b, t, units] x [input_dim, output_dim] = [b, t, output_dim]
            self.layer = tf.einsum('ijk,km->ijm', x, self.weights)
            if self.use_biases:
                self.layer = self.layer + self.biases
            return self.layer

    def call(self, x):
        return self.__call__(x)
