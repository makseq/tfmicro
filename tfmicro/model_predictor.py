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
import tensorflow as tf

from model_loader import Loader


# noinspection PyAttributeOutsideInit
class Predictor(Loader):

    def __init__(self, config):
        self.graph = tf.get_default_graph()
        try:
            self.prepare()
        except Exception:
            pass

    def prepare(self):
            self.input = self.graph.get_tensor_by_name("X:0")  # set input placeholder
            self.output = self.graph.get_tensor_by_name("output:0")  # set output operation

    def predict(self, x):
        result = self.sess.run(self.output, feed_dict={
            self.input: x,
        })
        return result

    def set_session(self, sess):
        self.sess = sess

    @classmethod
    def load(cls, path, forced_config=None):
        predictor = super(Predictor, cls).load(path, forced_config)

        predictor.graph = tf.get_default_graph()
        predictor.prepare()
        return predictor
