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
import sys
import time
import numpy as np
import tensorflow as tf
import gc

import threadgen
from model_loader import Loader
from model_predictor import Predictor


# noinspection PyAttributeOutsideInit
class Model(Loader):

    def progress(self, step):
        progress_width = 30
        step = self.data.steps_per_epoch if step == -1 else step
        dur = time.time() - self.epoch_time_start
        self.need_update = (time.time() - self.prev_time) > 0.5  # more 0.5 sec

        # costs
        train_cost_mean = np.mean(self.train_costs)
        train_cost_std = np.std(self.train_costs)
        test_cost_mean = np.mean(self.test_costs) if self.test_costs else 0
        test_cost_std = np.std(self.test_costs) if len(self.test_costs) > 1 else 0

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
        self.indicators = []
        self._reset_history()

    def add_indicator(self, reference, text):
        self.indicators += [{'reference': reference, 'text': text}]

    def _train_basics(self):
        c = self.c
        self.update_ops = []  # additional update params for graphs you need to call in session.run()
        self.learning_rate = c['model.optimizer.learning_rate']

        self.epoch_tf = tf.placeholder_with_default(tf.constant(-1, dtype=tf.int64), shape=[], name="epoch")
        self.step_tf = tf.placeholder_with_default(tf.constant(-1, dtype=tf.int64), shape=[], name="step")
        self.training = tf.placeholder_with_default(tf.constant(0, dtype=tf.int64), shape=[], name="is_training")
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

    def fit_data(self, data, callbacks=None, max_queue_size=100, thread_num=4, valid_thread_num=4,
                 tensorboard_subdir=''):
        c = self.c
        self.set_data(data)
        self.epochs = c['model.epochs']
        self.callbacks = [] if callbacks is None else callbacks
        self.train_generator = threadgen.ThreadedGenerator(data, 'train', max_queue_size, thread_num).start()
        self.valid_generator = threadgen.ThreadedGenerator(data, 'valid', max_queue_size, valid_thread_num).start()
        steps_per_epoch, validation_steps = data.steps_per_epoch, data.validation_steps

        # prepare train model
        print ' Compiling model'
        tf.reset_default_graph()
        self._train_model(data)

        # session init
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': c['use_gpu']}))
        self.sess.run(tf.global_variables_initializer())

        # log writer & model saver
        self.train_writer = tf.summary.FileWriter('./tensorboard/' + tensorboard_subdir + '/train')
        self.valid_writer = tf.summary.FileWriter('./tensorboard/' + tensorboard_subdir + '/valid')
        self.train_writer.add_graph(self.sess.graph)
        self.saver = tf.train.Saver()

        # load weights if we want to continue training
        if 'model.preload' in c and c['model.preload']:
            self.load_weights(c['model.preload'])

        # summary
        self.cost_summary = tf.summary.scalar("cost", self.cost)

        self._reset_history()
        self.epoch, self.step, train_cost, test_cost, restart = 1, 0, 0, 0, True
        self.epoch_time_start = time.time()
        self.train_costs, self.test_costs = [], []
        [call.set_model(self) for call in self.callbacks]  # set model to self.callbacks
        [call.set_config(c) for call in self.callbacks]  # set config to self.callbacks
        [call.on_start() for call in self.callbacks]  # self.callbacks
        print ' Train model'

        while self.epoch <= self.epochs:  # train cycle, we start from 1, so use <=
            ' epoch begin '
            if restart:
                restart = False
                self.step = 0
                self.info('\n  Epoch %i/%i\n' % (self.epoch, self.epochs))
                [call.on_epoch_begin() for call in self.callbacks]  # self.callbacks

            ' step begin '
            [call.on_step_begin() for call in self.callbacks]

            self.train_step()
            self.train_writer.flush()  # write summary to disk right now

            ' step end '
            [call.on_step_end() for call in self.callbacks]
            self.step += 1
            self.progress(self.step)  # print progress

            ' epoch end '
            if self.step >= steps_per_epoch or self.stop_training_now:

                ' validation pass '
                self.valid_step = 0
                while True:  # validation cycle
                    [call.on_validation_step_begin() for call in self.callbacks]
                    self.validation_step()
                    self.valid_writer.flush()  # write summary to disk right now
                    [call.on_validation_step_end() for call in self.callbacks]
                    self.progress(self.step)
                    self.valid_step += 1
                    if self.valid_step >= validation_steps:
                        break

                # print info to history
                self.history['loss'] += [np.mean(self.train_costs)]
                self.history['loss_std'] += [np.std(self.train_costs)]
                self.history['val_loss'] += [np.mean(self.test_costs)]
                self.history['val_loss_std'] += [np.std(self.test_costs)]
                self.history['lr'] += [self.learning_rate]
                self.history['time'] += [time.time() - self.epoch_time_start]
                self.train_costs, self.test_costs = [], []

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

    def _predict_model(self):
        if self.predictor is None:
            self.predictor = Predictor(self.c)
            self.predictor.set_session(self.sess)

    def predict(self, x):
        self._predict_model()
        return self.predictor.predict(x)

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
    def load(cls, path, forced_config=None):
        model = super(Model, cls).load(path, forced_config)
        model._reset_history()
        return model

    def load_weights(self, path):
        if os.path.isdir(path):  # path is dir
            c = json.load(open(path + '/config.json'))
        else:  # path is filename
            c = json.load(open(os.path.dirname(path) + '/config.json'))

        model_name = ''
        if os.path.isdir(path):  # take the last model
            models = set([m.split('.')[0].split('-')[1] for m in os.listdir(path) if 'model-' in m])  # get all models
            model_number = sorted([int(m) for m in models])[-1]  # last item
            model_name = '/model-%i' % model_number

        self.saver.restore(self.sess, path + model_name)
        print 'Variables loaded', path + model_name

        return
