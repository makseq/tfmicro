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
from tensorflow.core.framework import graph_pb2
from tensorflow.python.tools import freeze_graph


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

    @staticmethod
    def make_config_proto(c):
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

        return cfg

    @staticmethod
    def load_frozen_graph(frozen_path):
        """ Load frozen model
        
        :param frozen_path: path to frozen graph file
        """
        # read frozen graph
        graph_def = graph_pb2.GraphDef()
        with open(frozen_path, 'rb') as f:
            graph_def.ParseFromString(f.read())

        # import graph def
        tf.import_graph_def(graph_def, name='')
        print('Frozen model loaded:', frozen_path)

    @staticmethod
    def load_meta_graph(model, meta_graph, checkpoint):
        """ Load meta graph and weights as standard tensorflow model
        
        :param model: model class 
        :param meta_graph: meta graph path
        :param checkpoint: checkpoint class 
        """
        try:
            # import meta graph as usual
            model.saver = tf.train.import_meta_graph(meta_graph, clear_devices=True)
            print('Graph loaded:', meta_graph)
        except Exception as e:
            if not os.path.exists(meta_graph):
                print("No graph loaded! Path doesn't exist:", meta_graph)
            else:
                print('No graph loaded! Some errors occur:', meta_graph)
                print(e.__repr__())
            model.saver = tf.train.Saver()

        # load weights
        model.saver.restore(model.sess, checkpoint)
        print('Variables loaded:', checkpoint)

    @classmethod
    def load(cls, path, forced_config=None, print_vars=False, frozen_graph=True):
        """ Main model loading function
        
        :param path: path to tensorflow model root
        :param forced_config: dict as config
        :param print_vars: print debug info about loaded variables
        :param frozen_graph: freeze model graph and load it as frozen (best performance and memory consumption)
        :return: cls
        """
        # prepare config
        if forced_config is not None:
            c = forced_config
        else:
            # load config from file
            if os.path.isdir(path):  # path is dir
                c = json.load(open(path + '/config.json'))
            else:  # path is filename
                c = json.load(open(os.path.dirname(path) + '/config.json'))

        # get model filename
        checkpoint = tf.train.latest_checkpoint(path)
        meta_graph = checkpoint + '.meta'

        # reset
        tf.set_random_seed(1234)
        tf.reset_default_graph()

        # model setting up
        model = cls(c)
        model.c = c
        config_proto = cls.make_config_proto(c)  # prepare proto config for tensorflow

        # frozen graph loading
        if frozen_graph:
            frozen_path = os.path.join(path, 'frozen_graph.pb')

            # convert graph to frozen if no file
            if not os.path.exists(frozen_path):
                freeze_graph.freeze_graph(None, None, True, checkpoint, c['predict.output'].split(':')[0],
                                          None, None, frozen_path, True, None, input_meta_graph=meta_graph)
                print('Frozen model converted and saved:', frozen_path)

            # load frozen model
            tf.reset_default_graph()
            cls.load_frozen_graph(frozen_path)
            model.sess = tf.Session(config=config_proto)
            # FIXME: tensorflow bug: intra_op_parallelism_threads doesn't affects system right after freeze_graph()

        # regular tensorflow model loading
        else:
            model.sess = tf.Session(config=config_proto)
            cls.load_meta_graph(model, meta_graph, checkpoint)

        # print variables from loaded model
        if print_vars:
            [print(i) for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]

        cls.check_deprecated(c)
        return model
