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
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph


def make_config_proto(c):
    """ Read ConfigProto params from json. For example, put into c['tf.config.use_gpu'] = 1
    or c['tf.config.allow_growth'] = true, etc.

    :param c: json config
    :return: tf.ConfigProto
    """
    # tensorflow config
    use_gpu = os.environ.get('use_gpu', c.get('tf.config.use_gpu', c.get('use_gpu', 1)))
    cfg = tf.compat.v1.ConfigProto(device_count={'GPU': use_gpu})

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
    def load_frozen_graph(frozen_path):
        """ Load frozen model

        :param frozen_path: path to frozen graph file
        """
        # read frozen graph
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(frozen_path, 'rb') as f:
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
            model.saver = tf.compat.v1.train.import_meta_graph(meta_graph, clear_devices=True)
            print('Graph loaded:', meta_graph)
        except Exception as e:
            if not os.path.exists(meta_graph):
                print("No graph loaded! Path doesn't exist:", meta_graph)
            else:
                print('No graph loaded! Some errors occur:', meta_graph)
                print(e.__repr__())
            model.saver = tf.compat.v1.train.Saver()

        # load weights
        model.saver.restore(model.sess, checkpoint)
        print('Variables loaded:', checkpoint)

    @classmethod
    def load(cls, path, forced_config=None, print_vars=False, frozen_graph=True,
             tf_session_target='', tf_device=''):
        """ Main model loading function

        :param path: path to tensorflow model root
        :param forced_config: dict as config
        :param print_vars: print debug info about loaded variables
        :param frozen_graph: freeze model graph and load it as frozen (best performance and memory consumption)
        :param tf_session_target for tf server
        :param tf_device cuda device to use
        :return: cls
        """
        # prepare config
        if forced_config is not None:
            c = forced_config
            model_dir = path
        else:
            # load config from file
            if os.path.isdir(path):  # path is dir
                c = json.load(open(path + '/config.json'))
                model_dir = path

            # path is json file
            elif path.endswith('.json'):
                c = json.load(open(path))
                model_dir = os.path.dirname(path)

            # path is some filename
            else:
                c = json.load(open(os.path.dirname(path) + '/config.json'))
                model_dir = os.path.dirname(path)

        # get model filename
        checkpoint = tf.train.latest_checkpoint(model_dir)
        meta_graph = checkpoint + '.meta' if checkpoint is not None else 'no-checkpoint-found'

        # reset
        tf.compat.v1.set_random_seed(1234)
        tf.compat.v1.reset_default_graph()

        # model setting up
        model = cls(c)
        model.c = c
        config_proto = make_config_proto(c)  # prepare proto config for tensorflow

        target = tf_session_target if tf_session_target else c.get('tf.session.target', '')
        device = tf_device if tf_device else c.get('tf.device', '')
        if target or device:
            print('Model Loader: tf session target:', target, 'and device:', device)

        # frozen graph loading
        if frozen_graph and not c.get('tf.skip_frozen_graph', False):
            frozen_path = os.path.join(model_dir, 'frozen_graph.pb')

            # convert graph to frozen if no file
            if not os.path.exists(frozen_path):
                freeze_graph.freeze_graph(None, None, True, checkpoint, c['predict.output'].split(':')[0],
                                          None, None, frozen_path, True, None, input_meta_graph=meta_graph)
                print('Frozen model was converted and saved:', frozen_path)

            # load frozen model
            tf.compat.v1.reset_default_graph()
            with tf.device(device):
                cls.load_frozen_graph(frozen_path)

            model.sess = tf.compat.v1.Session(target=target, config=config_proto)
            # FIXME: tensorflow bug: intra_op_parallelism_threads doesn't affects system right after freeze_graph()

        # regular tensorflow model loading
        else:
            model.sess = tf.compat.v1.Session(target=target, config=config_proto)
            with tf.device(device):
                cls.load_meta_graph(model, meta_graph, checkpoint)

        # convert model to tflite format
        if c.get('tf.lite', False):
            tflite_model_path = os.path.join(model_dir, 'converted_model.tflite')
            if not os.path.exists(tflite_model_path):
                graph = tf.compat.v1.get_default_graph()
                input_tensor = graph.get_tensor_by_name(c['predict.input'])
                output_tensor = graph.get_tensor_by_name(c['predict.output'])
                print(input_tensor)
                print(output_tensor)
                timesteps = c.get('predict.split_size', c['model.timesteps'])
                dim = input_tensor.get_shape()[-1]
                input_tensor.set_shape([1, timesteps, dim])
                converter = tf.compat.v1.lite.TFLiteConverter.from_session(model.sess, [input_tensor], [output_tensor])
                tflite_model = converter.convert()
                with open(tflite_model_path, 'wb') as f:
                    f.write(tflite_model)
                print('Tflite model was converted and saved:', tflite_model_path)
            model.interpreter = tf.compat.v1.lite.Interpreter(model_path=tflite_model_path)
            model.interpreter.allocate_tensors()
            print('Tflite model was loaded:', tflite_model_path)



        # print variables from loaded model
        if print_vars:
            [print(i) for i in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)]
            print('--- Operations ---')
            [print(n.name) for n in tf.compat.v1.get_default_graph().as_graph_def().node]

        cls.check_deprecated(c)
        return model


# noinspection PyAttributeOutsideInit
class Predictor(Loader):

    def __init__(self, config):
        self.graph = tf.compat.v1.get_default_graph()
        self.config = config

    def prepare(self):
        """ Override this function in your own predictor class if need """

        c = self.config
        if c.get('tf.lite', False):
            # Get input and output tensors.
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            inp = self.config.get('predict.input', 'X:0')
            out = self.config.get('predict.output', 'output:0')

            # set input placeholder
            self.input = self.graph.get_tensor_by_name(inp)

            # set output operation
            if isinstance(out, str):
                self.output = {out: self.graph.get_tensor_by_name(out)}
                self.output_alone = True
            elif isinstance(out, list):
                self.output = {o: self.graph.get_tensor_by_name(o) for o in out}
                self.output_alone = False
            else:
                raise LoaderException('incorrect predict.output type')

    def predict(self, feats):
        c = self.config
        if c.get('tf.lite', False):
            # Test the model on random input data.
            input_shape = self.input_details[0]['shape']
            output_shape = self.output_details[0]['shape']
            self.interpreter.set_tensor(self.input_details[0]['index'], feats)

            self.interpreter.invoke()

            # The function `get_tensor()` returns a copy of the tensor data.
            # Use `tensor()` in order to get a pointer to the tensor.
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            return output_data
        else:
            results = self.sess.run(list(self.output.values()), feed_dict={
                self.input: feats
            })
            self.full_results = {key: results[i] for i, key in enumerate(self.output.keys())}

            if self.output_alone:
                return results[0]
            else:
                return self.full_results

    def split_predict(self, feats, batch_size, feature_steps, zero):
        """ Split features x into parts with equal size and feed it into predict

        :param feats: features
        :return: averaged vectors from splitted dvectors
        """
        # split long features into parts by timesteps
        parts = [feats[i:i + feature_steps] for i in range(0, len(feats), feature_steps)]  # N x timesteps x dim
        parts = np.array(parts)
        dim = feats.shape[-1]
        d_parts = []

        # reformat batch with created parts
        for b in range(0, len(parts), batch_size):
            # prepare batch
            p = parts[b:b + batch_size]  # new batch: len(p), timesteps, ?
            x = np.zeros((len(p), feature_steps, dim), dtype=np.float32)

            for i in range(len(p)):  # timesteps can be variable in parts
                x[i, :] = zero
                x[i, 0:len(p[i]), :] = p[i]

            # evaluate dvector
            d = self.predict(x)[0:len(p)]
            d_parts += [d]

        # mean all dvector parts
        d_parts = np.vstack(d_parts)
        self.d_parts = d_parts
        dvector = np.mean(d_parts, axis=0)
        print(len(parts), dvector.shape, np.mean(dvector), np.std(dvector))
        return dvector

    def set_session(self, sess):
        self.sess = sess

    @classmethod
    def load(cls, path, forced_config=None, *args, **kwargs):
        predictor = super(Predictor, cls).load(path, forced_config, *args, **kwargs)

        predictor.graph = tf.compat.v1.get_default_graph()
        predictor.prepare()
        return predictor
