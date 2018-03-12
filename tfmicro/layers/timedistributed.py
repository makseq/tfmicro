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

    def call(self, x):
        # [b, t, units] x [input_dim, output_dim] = [b, t, output_dim]
        self.layer = tf.einsum('ijk,km->ijm', x, self.weights)
        if self.use_biases:
            self.layer = self.layer + self.biases
        return self.layer
