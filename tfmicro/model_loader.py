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
import json
import os
import tensorflow as tf


class Loader(object):

    @classmethod
    def load(cls, path, forced_config=None):
        # prepare config
        if forced_config is not None:
            c = forced_config
        else:
            # load config from file
            if os.path.isdir(path):  # path is dir
                c = json.load(open(path + '/config.json'))
            else:  # path is filename
                c = json.load(open(os.path.dirname(path) + '/config.json'))

        model = cls(c)
        tf.reset_default_graph()
        tf.set_random_seed(1234)

        use_gpu = c['use_gpu'] if 'use_gpu' not in os.environ else os.environ['use_gpu']
        allow_growth = c['allow_growth'] if 'allow_growth' in c else True

        config_proto = tf.ConfigProto(device_count={'GPU': c['use_gpu']})
        config_proto.gpu_options.allow_growth = True
        model.sess = tf.Session(config=config_proto)

        model_name = ''
        if os.path.isdir(path):  # take the last model
            models = set([m.split('.')[0].split('-')[1] for m in os.listdir(path) if 'model-' in m])  # get all models
            model_number = sorted([int(m) for m in models])[-1]  # last item
            model_name = '/model-%i' % model_number

        try:
            graph_path = path + model_name + '.meta'
            model.saver = tf.train.import_meta_graph(graph_path, clear_devices=True)
            print 'Graph loaded', graph_path
        except Exception as e:
            if not os.path.exists(graph_path):
                print "No graph loaded! Path doesn't exist:", graph_path
            else:
                print 'No graph loaded! Some errors occur:', graph_path
                print e.__repr__()
            model.saver = tf.train.Saver()

        model.saver.restore(model.sess, path + model_name)
        print 'Variables loaded', path + model_name
        model.c = c
        return model

