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
from __future__ import print_function
import json
import os
import tensorflow as tf


class LoaderException(Exception):
    pass


class Loader(object):

    @staticmethod
    def check_deprecated(config):
        if 'use_gpu' in config:
            print("\n! warning: 'use_gpu' in config is deprecated. Use 'tf.config.use_gpu' instead.")

        if 'allow_growth' in config:
            print("\n! warning: 'allow_growth' in config is deprecated. "
                  "Use 'tf.config.gpu_options.allow_growth' instead.")

    @classmethod
    def load(cls, path, forced_config=None, print_vars=False):
        # prepare config
        if forced_config is not None:
            c = forced_config
        else:
            # load config from file
            if os.path.isdir(path):  # path is dir
                c = json.load(open(path + '/config.json'))
            else:  # path is filename
                c = json.load(open(os.path.dirname(path) + '/config.json'))

        # init & reset
        model = cls(c)
        tf.reset_default_graph()
        tf.set_random_seed(1234)

        # tensorflow config
        use_gpu = os.environ.get('use_gpu', c.get('tf.config.use_gpu', c.get('use_gpu', 1)))
        cfg = tf.ConfigProto(device_count={'GPU': use_gpu})

        # scan config for tf config params
        opts = {opt.replace('tf.config.', ''): c[opt] for opt in c if opt.startswith('tf.config.')}
        if 'use_gpu' in opts:
            del opts['use_gpu']  # it's not standard {key: value} parameter

        # apply params to config proto
        for key, value in opts.items():
            if '.' in key:  # go deeper
                split = key.split('.')
                if len(split) > 2:
                    raise Exception('Only two level config supported: ' + split)

                key1, key2 = split[0], split[1]
                root_obj = getattr(cfg, key1)
                setattr(root_obj, key2, value)
            else:
                setattr(cfg, key, value)

        model.sess = tf.Session(config=cfg)

        # get model filename
        model_name = ''
        if os.path.isdir(path):  # take the last model
            models = set([m.split('.')[0].split('-')[1] for m in os.listdir(path) if 'model-' in m])  # get all models
            model_number = sorted([int(m) for m in models])[-1]  # take last model from list
            model_name = '/model-%i' % model_number
        graph_path = path + model_name + '.meta'

        # import meta graph
        try:
            model.saver = tf.train.import_meta_graph(graph_path, clear_devices=True)
            print('Graph loaded', graph_path)
        except Exception as e:
            if not os.path.exists(graph_path):
                print("No graph loaded! Path doesn't exist:", graph_path)
            else:
                print('No graph loaded! Some errors occur:', graph_path)
                print(e.__repr__())
            model.saver = tf.train.Saver()

        # print variables from loaded model
        if print_vars:
            for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                print(i)

        # load weights
        model.saver.restore(model.sess, path + model_name)
        print('Variables loaded', path + model_name)
        model.c = c

        cls.check_deprecated(c)
        return model

