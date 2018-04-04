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

from model_loader import Loader


# noinspection PyAttributeOutsideInit
class Predictor(Loader):

    def __init__(self, config):
        self.graph = tf.get_default_graph()
        self.config = config

    def prepare(self):
        """ Override this function in your own predictor class if need """

        inp = self.config['predict.input'] if 'predict.input' in self.config else "X:0"
        out = self.config['predict.output'] if 'predict.output' in self.config else "output:0"

        self.input = self.graph.get_tensor_by_name(inp)  # set input placeholder
        self.output = self.graph.get_tensor_by_name(out)  # set output operation

    def predict(self, feats):
        result = self.sess.run(self.output, feed_dict={
            self.input: feats
        })
        return result

    def split_predict(self, feats, batch_size, feature_steps, zero):
        """ Split features x into parts with equal size and feed it into predict  
            
        :param feats: features
        :return: averaged vectors from splitted dvectors
        """

        # split long features into parts by timesteps
        parts = [feats[i:i + feature_steps] for i in xrange(0, len(feats), feature_steps)]  # N x timesteps x dim
        parts = np.array(parts)
        dim = feats.shape[-1]
        d_parts = []

        # reformat batch with created parts
        for b in xrange(0, len(parts), batch_size):
            # prepare batch
            p = parts[b:b + batch_size]  # new batch: len(p), timesteps, ?
            x = np.zeros((len(p), feature_steps, dim))

            for i in xrange(len(p)):  # timesteps can be variable in parts
                x[i, :] = zero
                x[i, 0:len(p[i]), :] = p[i]

            # evaluate dvector
            d = self.predict(x)[0:len(p)]
            d_parts += [d]

        # mean all dvector parts
        d_parts = np.vstack(d_parts)
        dvector = np.mean(d_parts, axis=0)
        return dvector

    def set_session(self, sess):
        self.sess = sess

    @classmethod
    def load(cls, path, forced_config=None):
        predictor = super(Predictor, cls).load(path, forced_config)

        predictor.graph = tf.get_default_graph()
        predictor.prepare()
        return predictor
