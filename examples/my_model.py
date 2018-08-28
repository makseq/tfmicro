'''
 This is template only, but NOT WORKING example. 
'''

import tensorflow as tf
from tfmicro import model

# noinspection PyAttributeOutsideInit
class MyModel(model.Model):

    def _train_model(self, data):
        c = self.c
        self._train_basics()

        # set indicator adam/sgd
        self.add_indicator(lambda: 0.5, lambda: 'text indicator')

        batch_size = data.batch_size
        units = c['model.units']
        self.alfa = c['model.alfa']

        ### PLACE YOUR MODEL HERE ###
        self.X = tf.placeholder(tf.float32, shape=[batch_size, data.num_files,  data.timesteps, data.dim], name="X")
        cell_fw = tf.nn.rnn_cell.GRUCell(units[0])
        self.output, _ = tf.nn.dynamic_rnn(cell_fw, self.X, dtype=tf.float32)  # => [batch, time, units[0]]
        self.out = self.output

        # cost & optimizer
        with tf.name_scope("cost_optimizer"):
            # loss function
            if c['model.loss'] == 'none':
                self.cost = tf.reduce_mean(self.output, axis=[0,1,2])

            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_tf).minimize(self.cost)

    def train_step(self):
        c = self.c
        self.x, self.y = self.train_generator.get_values()

        # train step
        params = [self.cost, self.summary, self.optimizer, self.out] + self.update_ops
        out = self.sess.run(params, feed_dict={
            self.X: self.x,

            self.training: 1,
            self.step_tf: self.step,
            self.epoch_tf: self.epoch,
            self.learning_rate_tf: self.learning_rate
        })
        cost, summary, _, self.train_prediction = out[0:-len(self.update_ops)] if self.update_ops else out

        # print context

        self.train_writer.add_summary(summary, global_step=self.epoch)
        if self.step == 0:
            self.train_writer.add_summary(out[-2], global_step=self.epoch)
            self.train_writer.add_summary(out[-1], global_step=self.epoch)
        self.train_costs += [cost]

    def validation_step(self):
        self.test_x, self.test_y = self.valid_generator.get_values()

        params = [self.cost_summary, self.cost, self.out] + self.update_ops
        out = self.sess.run(params, feed_dict={self.X: self.test_x})
        cost_summary, cost, self.test_prediction = out[0:-len(self.update_ops)] if self.update_ops else out

        self.test_costs += [cost]
        self.test_writer.add_summary(cost_summary, global_step=self.epoch)