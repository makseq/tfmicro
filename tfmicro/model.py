import json
import os
import sys
import time
import numpy as np
import tensorflow as tf
import gc

import threadgen


class Predictor(object):

    def __init__(self, train_model):
        graph = tf.get_default_graph()
        self.X = graph.get_tensor_by_name("X:0")  # set input
        self.Y = graph.get_tensor_by_name("Y:0")  # set input
        self.output = graph.get_tensor_by_name("output:0")  # set output operation
        self.sess = train_model.sess
        self.train_model = train_model
        self.update_ops = []

    def set_update_ops(self, ops):
        self.update_ops = ops

    def predict(self, x):
        out_shape = self.Y.get_shape()
        params = [self.output] + self.update_ops
        result = self.sess.run(params, feed_dict={
            self.X: x,
            self.Y: np.zeros([len(x), out_shape[1], out_shape[2]])
        })
        return result[0]


# noinspection PyAttributeOutsideInit
class Model(object):

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

        p = eval_progress(step)
        self.need_update = self.need_update or (p != eval_progress(step - 1) and step >= 0)
        self.need_update = self.need_update or step == self.data.steps_per_epoch
        if self.need_update:
            self.prev_time = time.time()

            msg1 = '  [' + '=' * p + '-' * (progress_width - p) + '] %i/%i' % (step, self.data.steps_per_epoch)
            msgt = ' %0.0f/%0.0fs' % (dur, (dur/float(step)) * self.data.steps_per_epoch) if step > 0 else ''
            msg2 = ' > train: %0.4f [%0.2f]' % (train_cost_mean, train_cost_std)
            msg3 = ' > test: %0.4f ' % test_cost_mean if test_cost_mean > 0 else ''
            msg4 = '[%0.2f]' % test_cost_std if len(self.test_costs) > 1 else ''

            if step == 0:
                # sys.stdout.write(' ' + msg1 + '\n' + msgt + msg2 + msg3)
                sys.stdout.write(' ' + msg1 + msgt + msg2 + msg3 + msg4)
            else:
                # sys.stdout.write('\033[F\r' + msg1 + msgt + '\n\033[K' + msg2 + msg3)
                sys.stdout.write('\033[K\r' + msg1 + msgt + msg2 + msg3 + msg4)
            sys.stdout.flush()


    def __init__(self, c):
        self.c = c
        self.stop_training = False
        self.stop_training_now = False
        self.predictor = None
        self.prev_time = time.time()

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

    def fit_data(self, data, callbacks=None, epochs=100, max_queue_size=100, thread_num=4, valid_thread_num=4, use_gpu=True,
                 tensorboard_subdir=''):
        c = self.c
        self.set_data(data)
        self.epochs = epochs
        self.callbacks = [] if callbacks is None else callbacks
        self.train_generator = threadgen.ThreadedGenerator(data.generator('train'), max_queue_size, thread_num).start()
        self.valid_generator = threadgen.ThreadedGenerator(data.generator('valid'), max_queue_size, valid_thread_num).start()
        steps_per_epoch, validation_steps = data.steps_per_epoch, data.validation_steps

        # prepare train model
        print ' Compiling model'
        tf.reset_default_graph()
        self._train_model(data)

        # session init
        self.sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': use_gpu}))
        self.sess.run(tf.global_variables_initializer())

        # log writer & model saver
        self.train_writer = tf.summary.FileWriter('./tensorboard/' + tensorboard_subdir + '/train')
        self.valid_writer = tf.summary.FileWriter('./tensorboard/' + tensorboard_subdir + '/valid')
        self.train_writer.add_graph(self.sess.graph)
        self.saver = tf.train.Saver()

        # summary
        self.cost_summary = tf.summary.scalar("cost", self.cost)

        self.history = {'loss': [], 'val_loss': [], 'loss_std': [], 'val_loss_std': [], 'time': [], 'lr': []}
        self.epoch, self.step, train_cost, test_cost, first = 1, 0, 0, 0, True
        self.epoch_time_start = time.time()
        self.train_costs, self.test_costs = [], []
        [call.set_model(self) for call in self.callbacks]  # set model to self.callbacks
        [call.set_config(c) for call in self.callbacks]  # set config to self.callbacks
        [call.on_start() for call in self.callbacks]  # self.callbacks
        print ' Train model'

        while self.epoch <= self.epochs:  # train cycle, we start from 1, so use <=
            ' epoch begin '
            if first:
                first = False
                self.step = 0
                sys.stdout.write('\n  Epoch %i/%i\n' % (self.epoch, self.epochs))
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
                first = True
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
            self.predictor = Predictor(self)

    def predict(self, x):
        self._predict_model()
        return self.predictor.predict(x)

    def set_data(self, data):
        self.data = data

    def save(self, dir_path, saver=None):
        saver = self.saver if saver is None else saver
        os.makedirs(dir_path) if not os.path.exists(dir_path) else ()
        saver.save(self.sess, dir_path + '/model', global_step=self.epoch)
        json.dump(self.c, open(dir_path + '/config.json', 'w'), indent=4)

    @classmethod
    def load(cls, path):
        if os.path.isdir(path):  # path is dir
            c = json.load(open(path + '/config.json'))
        else:  # path is filename
            c = json.load(open(os.path.dirname(path) + '/config.json'))

        model = cls(c)
        tf.reset_default_graph()
        model.sess = tf.Session()

        model_name = ''
        if os.path.isdir(path):  # take the last model
            models = set([m.split('.')[0].split('-')[1] for m in os.listdir(path) if 'model-' in m])  # get all models
            model_number = sorted([int(m) for m in models])[-1]  # last item
            model_name = '/model-%i' % model_number
            
        try:
            graph_path = path + model_name + '.meta'
            model.saver = tf.train.import_meta_graph(graph_path)
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