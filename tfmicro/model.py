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
import gc
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python import pywrap_tensorflow

from . import threadgen
from . import keyboard
from .model_predictor import Loader, Predictor, make_config_proto


# noinspection PyAttributeOutsideInit
class Model(Loader):

    def progress(self, step):
        progress_width = 30
        step = self.data.steps_per_epoch if step == -1 else step
        dur = time.time() - self.epoch_time_start
        self.need_update = (time.time() - self.prev_time) > 0.5  # more 0.5 sec

        # costs
        train_cost_mean = np.mean(self.train_costs) if self.train_costs else 0
        train_cost_std = np.std(self.train_costs) if self.train_costs else 0
        test_cost_mean = np.mean(self.test_costs) if self.test_costs else 0
        test_cost_std = np.std(self.test_costs) if self.test_costs else 0

        def eval_progress(s):
            return int(s / float(self.data.steps_per_epoch) * progress_width + 0.5)

        def eval_indicator(s):
            return int(s * progress_width + 0.5)

        p = eval_progress(step)
        self.need_update = self.need_update or (p != eval_progress(step - 1) and step >= 0)
        self.need_update = self.need_update or step == self.data.steps_per_epoch
        if self.need_update:
            self.prev_time = time.time()

            msg1 = '  [' + '=' * p + '-' * (progress_width - p) + '] %i/%i' % (step, self.data.steps_per_epoch)
            msgt = ' %0.0f/%0.0fs' % (dur, (dur/float(step)) * self.data.steps_per_epoch) if step > 0 else ''
            msg2 = ' > train: %0.4f [%0.2f]' % (train_cost_mean, train_cost_std)
            msg3 = ' > valid: %0.4f ' % test_cost_mean if test_cost_mean > 0 else ''
            msg4 = '[%0.2f]' % test_cost_std if len(self.test_costs) > 1 else ''

            # back line for indicators
            [sys.stdout.write('\033[F') for _ in self.indicators]

            sys.stdout.write('\033[K\r' + msg1 + msgt + msg2 + msg3 + msg4)

            # print indicators progress
            for indicator in self.indicators:
                p = eval_indicator(indicator['reference']())
                msg_ind = '  [' + '=' * p + ' ' * (progress_width - p) + '] %s' % indicator['text']()
                sys.stdout.write('\n\033[K' + msg_ind)

            sys.stdout.flush()

    def info(self, args, return_lines=True):
        for a in args:
            sys.stdout.write(a)
        if return_lines:
            for _ in self.indicators:
                sys.stdout.write('\n')

    def __init__(self, config):
        self.c = config
        self.stop_training = False
        self.stop_training_now = False
        self.predictor = None
        self.prev_time = time.time()
        self.history = None
        self.epoch = 0
        self.indicators = []
        self.sess = None
        self.saver = None
        self.train_writer = None
        self.valid_writer = None
        self._reset_history()
        self.tensorboard_root = './tensorboard/'

    def add_indicator(self, reference, text):
        self.indicators += [{'reference': reference, 'text': text}]

    def _train_basics(self):
        c = self.c
        self.update_ops = []  # additional update params for graphs you need to call in session.run()
        self.learning_rate = c['model.optimizer.learning_rate']

        self.epoch_tf = tf.placeholder_with_default(tf.constant(-1, dtype=tf.int64), shape=[], name="epoch")
        self.step_tf = tf.placeholder_with_default(tf.constant(-1, dtype=tf.int64), shape=[], name="step")
        self.training = tf.placeholder_with_default(tf.constant(0, dtype=tf.int64), shape=[], name="training")
        self.is_training = tf.equal(self.training, 1)
        self.learning_rate_tf = tf.placeholder_with_default(tf.constant(self.learning_rate, dtype=tf.float32), shape=[])

    def _train_model(self, data):
        c = self.c
        self._train_basics()  # prepare basic placeholders

        batch_size = data.batch_size
        self.X = tf.placeholder(tf.float32, shape=[batch_size, data.input_len,  data.input_dim], name="X")
        self.Y = tf.placeholder(tf.float32, shape=[batch_size, data.output_len, data.output_dim], name="Y")

        layer = self.X

        # ... your model here ...

        # Output matmul
        units = c['model.units']
        weights = tf.Variable(tf.truncated_normal([units, data.output_dim], stddev=0.5))
        bias = tf.Variable(tf.constant(0.1, shape=[data.output_dim]))

        shape = tf.shape(layer)
        layer = tf.reshape(layer, [shape[0] * data.output_len, units])
        layer = tf.matmul(layer, weights) + bias  # [m, units] x [n, output_dim] = [m, output_dim]
        layer = tf.reshape(layer, [shape[0], data.output_len, data.output_dim])

        out = tf.identity(layer, name="output")
        self.out = out

        # cost & optimizer
        with tf.name_scope("cost_optimizer"):
            # loss function
            diff = tf.reduce_mean(tf.square(self.Y - out))
            self.cost = tf.clip_by_value(diff, 1e-40, 1e10)

            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_tf).minimize(self.cost)

    def train_step(self):
        # get data
        self.x, self.y = self.train_generator.get_values()

        # train step
        params = [self.cost, self.cost_summary, self.optimizer, self.out] + self.update_ops
        cost, cost_summary, _, self.train_prediction = self.sess.run(params, feed_dict={
            self.X: self.x,
            self.Y: self.y,

            self.training: 1,
            self.step_tf: self.step,
            self.epoch_tf: self.epoch,
            self.learning_rate_tf: self.learning_rate
        })

        self.train_writer.add_summary(cost_summary, global_step=self.epoch*self.data.steps_per_epoch + self.step)
        self.train_costs += [cost]

    def validation_step(self):
        # get data
        self.test_x, self.test_y = self.valid_generator.get_values()

        # validate
        params = [self.cost, self.cost_summary, self.out] + self.update_ops
        cost, cost_summary, self.test_prediction = self.sess.run(params, feed_dict={
                                                    self.X: self.test_x, self.Y: self.test_y})

        self.valid_writer.add_summary(cost_summary, global_step=self.epoch*self.data.validation_steps + self.valid_step)
        self.test_costs += [cost]

    def _reset_history(self):
        self.history = {'loss': [], 'val_loss': [], 'loss_std': [], 'val_loss_std': [], 'time': [], 'lr': []}

    def run_validation(self, write_history=True, run_callbacks=True):
        self.valid_step = 0
        self.test_costs = []
        [call.on_validation_begin() for call in self.callbacks if run_callbacks]

        while True:  # validation cycle
            [call.on_validation_step_begin() for call in self.callbacks if run_callbacks]
            self.validation_step()
            self.valid_writer.flush()  # write summary to disk right now
            [call.on_validation_step_end() for call in self.callbacks if run_callbacks]
            self.progress(self.step)
            self.valid_step += 1
            if self.valid_step >= self.data.validation_steps:
                break

        # print info to history
        if write_history:
            self.history['loss'] += [np.mean(self.train_costs)]
            self.history['loss_std'] += [np.std(self.train_costs)]
            self.history['val_loss'] += [np.mean(self.test_costs)]
            self.history['val_loss_std'] += [np.std(self.test_costs)]
            self.history['lr'] += [self.learning_rate]
            self.history['time'] += [time.time() - self.epoch_time_start]
            self.train_costs, self.test_costs = [], []

        [call.on_validation_end() for call in self.callbacks if run_callbacks]

    def fit_data(self, data, callbacks=None, max_queue_size=100, thread_num=4, valid_thread_num=4,
                 tensorboard_subdir=''):
        c = self.c
        # check deprecated function
        self.check_deprecated(c)

        self.set_data(data)
        self.epochs = c['model.epochs']
        self.callbacks = [] if callbacks is None else callbacks
        self.train_generator = threadgen.ThreadedGenerator(data, 'train', max_queue_size, thread_num).start()
        self.valid_generator = threadgen.ThreadedGenerator(data, 'valid', max_queue_size, valid_thread_num).start()
        self.keyboard = keyboard
        self.keyboard.start()

        # prepare train model
        device = c.get('model.tf.device', '')
        with tf.device(device):
            print(' Compiling model' + (' for device ' + device) if device else '')
            tf.reset_default_graph()
            tf.set_random_seed(1234)
            self._train_model(data)

        # session init & tf_debug
        if c.get('tf.session.target', ''):
            print('model tf session target:', c.get('tf.session.target', ''))
        self.sess = tf.Session(target=c.get('tf.session.target', ''), config=make_config_proto(c))
        if self.c.get('tf.debug.enabled', False):
            port = self.c.get('tf.debug.port', '6064')
            self.sess = tf_debug.TensorBoardDebugWrapperSession(self.sess, 'localhost:'+port)
        self.sess.run(tf.global_variables_initializer())

        # log writer & model saver
        self.tensorboard_subdir = os.path.join(self.tensorboard_root, tensorboard_subdir)
        with tf.summary.FileWriter(self.tensorboard_subdir + '/train') as self.train_writer, \
             tf.summary.FileWriter(self.tensorboard_subdir + '/valid') as self.valid_writer:

            self.train_writer.add_graph(self.sess.graph)
            if self.saver is None:
                self.saver = tf.train.Saver()

            # load weights if we want to continue training
            if 'model.preload' in c and c['model.preload']:
                self.load_weights(c['model.preload'], c.get('model.preload.verbose', False))

            # summary
            self.cost_summary = tf.summary.scalar("cost", self.cost)

            self._reset_history()
            self.epoch, self.step, train_cost, test_cost, restart = 1, 0, 0, 0, True
            self.epoch_time_start = time.time()
            self.train_costs, self.test_costs = [], []
            [call.set_model(self) for call in self.callbacks]  # set model to self.callbacks
            [call.set_config(c) for call in self.callbacks]  # set config to self.callbacks
            [call.on_start() for call in self.callbacks]  # self.callbacks
            print(' Train model')

            while self.epoch <= self.epochs:  # train cycle, we start from 1, so use <=
                ' epoch begin '
                if restart:
                    restart = False
                    self.step = 0
                    self.info('\n  Epoch %i/%i\n' % (self.epoch, self.epochs))
                    [call.on_epoch_begin() for call in self.callbacks]  # self.callbacks

                ' step begin '
                print(self.epoch, 'sldkfjalsd')
                [call.on_step_begin() for call in self.callbacks]
                print(self.epoch, 'sldkfjalsd')

                self.train_step()
                print(self.epoch, 'sldkfjalsd')
                self.train_writer.flush()  # write summary to disk right now
                print(self.epoch, 'sldkfjalsd')

                ' step end '
                [call.on_step_end() for call in self.callbacks]
                self.step += 1
                print(self.epoch, 'sldkfjalsd')
                self.progress(self.step)  # print progress
                print(self.epoch, 'sldkfjalsd')

                ' epoch end '
                if self.step >= self.data.steps_per_epoch or self.stop_training_now:

                    ' validation pass '
                    self.run_validation()

                    # self.callbacks: on epoch end
                    [call.on_epoch_end() for call in self.callbacks]
                    sys.stdout.write('\n')

                    # reset & stop check
                    restart = True
                    self.epoch += 1
                    self.epoch_time_start = time.time()
                    if self.stop_training or self.stop_training_now:
                        break  # break main loop

            self.train_generator.stop()
            self.valid_generator.stop()
            [call.on_finish() for call in self.callbacks]  # self.callbacks
            gc.collect()
            return self

    def get_predictor(self, predictor_cls):
        if self.predictor is None:
            self.predictor = predictor_cls(self.c)
            self.predictor.prepare()
            self.predictor.set_session(self.sess)
        return self.predictor

    def set_data(self, data):
        self.data = data

    def set_config(self, config):
        self.c = config

    def save(self, dir_path, saver=None):
        saver = self.saver if saver is None else saver
        os.makedirs(dir_path) if not os.path.exists(dir_path) else ()
        saver.save(self.sess, dir_path + '/model', global_step=self.epoch)
        json.dump(self.c, open(dir_path + '/config.json', 'w'), indent=4)

    @classmethod
    def load(cls, path, forced_config=None, *args, **kwargs):
        model = super(Model, cls).load(path, forced_config, *args, **kwargs)
        model._reset_history()
        return model

    def load_weights(self, path, verbose=False):
        """ Load weights to current graph. 
        It loads only variables with the same names and shapes from the checkpoint. 
        
        :param path: path to model 
        :param verbose: print debug info if True
        :return: None
        """
        if 'model.preload.verbose' in self.c:
            verbose = self.c['model.preload.verbose']

        if os.path.isdir(path):  # path is dir
            c = json.load(open(path + '/config.json'))
        else:  # path is filename
            c = json.load(open(os.path.dirname(path) + '/config.json'))

        model_name = ''
        if os.path.isdir(path):  # take the last model
            models = set([m.split('.')[0].split('-')[1] for m in os.listdir(path) if 'model-' in m])  # get all models
            model_number = sorted([int(m) for m in models])[-1]  # last item
            model_name = '/model-%i' % model_number

        # get variables from _train_model (current graph)
        current_vars = current_vars_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # exclude variables using config exclude_var_names
        if 'model.preload.exclude_var_names' in self.c:
            new = []
            exclude_names = self.c['model.preload.exclude_var_names']
            if not isinstance(exclude_names, list):
                raise Exception('model.preload.exclude_var_names must be list of strings')

            for v in current_vars_all:
                exclude = [True for substr in exclude_names if substr in v.name]
                if not exclude:
                    new += [v]
            current_vars = new

        # get variable names from checkpoint
        reader = pywrap_tensorflow.NewCheckpointReader(path + model_name)
        loading_shapes = reader.get_variable_to_shape_map()
        loading_names = sorted(reader.get_variable_to_shape_map())

        # find intersect of loading and current variables
        intersect_vars = []
        ignored_names = []
        for n in loading_names:
            included = False
            # add var
            for v in current_vars:
                if n == v.name.split(':')[0] and v.shape == loading_shapes[n]:
                    intersect_vars += [v]
                    included = True
            # ignore var
            if not included:
                ignored_names += [n]

        # print intersection
        if verbose:
            if 'model.preload.exclude_var_names' in self.c:
                print('\nExcluded variables:')
                print(self.c['model.preload.exclude_var_names'])

            print('\nVariables from current model - exclude_var_names (from config):')
            for i in sorted([v.name for v in current_vars]):
                print(' ', i)

            print('\nVariables from loading model:')
            for key in loading_names:
                print(' ', key)

            print('\nIntersect variables:')
            for v in intersect_vars:
                print(' ', v.name)

            print('\nIgnored variables:')
            for n in ignored_names:
                print(' ', n)
            print()
        saver = tf.train.Saver(var_list=intersect_vars)
        saver.restore(self.sess, path + model_name)
        print(' ', str(len(intersect_vars)) + '/' + str(len(current_vars_all)), 'variables loaded', path + model_name,
              '\n')
        return

    @staticmethod
    def get_parameter_number(scope):
        """ Get variable parameters number from scope
        """
        total_parameters = 0
        for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope):
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

    def summary_image(self, image, name):
        """ Attach a image of a Tensor (for TensorBoard visualization)
        
        :param image: image tensor with shape rank is 2 or 3  
        :param name: name for tensorboard
        """
        if len(image.get_shape()) == 2:
            image = tf.expand_dims(image, -1)
            image = tf.expand_dims(image, 0)
        elif len(image.get_shape()) == 3:
            image = tf.expand_dims(image, -1)

        self.summaries.append(tf.summary.image(name, image))

    def summary_tensor(self, tensor, name):
        """ Attach mean, std, max, min, histogram of summaries to a Tensor for tensor (for TensorBoard visualization).
         
        :param tensor: tensor
        :param name: name for tensorboard
        """
        with tf.name_scope(name):
            mean = tf.reduce_mean(tensor)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
            self.summaries.append(tf.summary.scalar('mean', mean))
            self.summaries.append(tf.summary.scalar('stddev', stddev))
            self.summaries.append(tf.summary.scalar('max', tf.reduce_max(tensor)))
            self.summaries.append(tf.summary.scalar('min', tf.reduce_min(tensor)))
            self.summaries.append(tf.summary.histogram('histogram', tensor))

    def check_deprecated(self, config):
        if 'use_gpu' in config:
            print("\n! warning: 'use_gpu' in config is deprecated. Use 'tf.config.use_gpu' instead.")

        if 'allow_growth' in config:
            print("\n! warning: 'allow_growth' in config is deprecated. "
                  "Use 'tf.config.gpu_options.allow_growth' instead.")

        if hasattr(self, 'variable_image'):
            print("\n! warning: you use 'variable_image' in your model, "
                  "but tfmicro has built-in method 'tfmicro.Model.summary_image'")

        if hasattr(self, 'variable_summaries'):
            print("\n! warning: you use 'variable_summaries' in your model, "
                  "but tfmicro has built-in method 'tfmicro.Model.summary_tensor'")
