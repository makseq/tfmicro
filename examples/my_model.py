import tensorflow as tf
from tfmicro import model


# noinspection PyAttributeOutsideInit
class MyModel(model.Model):

    def _predict_model(self):
        model.Model._predict_model(self)

        # updates for stateful states
        if self.c['model.stateful']:
            self.predictor.set_update_ops([
                tf.get_default_graph().get_tensor_by_name("encoder/update_ops1:0"),
                tf.get_default_graph().get_tensor_by_name("encoder/update_ops2:0")
            ])

    @staticmethod
    def time_distributed_dense(layer, input_shape, output_dim):
        # Output matmul
        bias = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        weights = tf.Variable(tf.truncated_normal([input_shape[-1], output_dim], stddev=0.5, dtype=tf.float32))

        layer = tf.reshape(layer, [input_shape[0] * input_shape[1], input_shape[2]])
        layer = tf.matmul(layer, weights) + bias  # [m, units] x [n, output_dim] = [m, output_dim]
        layer = tf.reshape(layer, [input_shape[0], input_shape[1], output_dim])  # data.output_dim
        return layer

    def _train_model(self, data):
        c = self.c
        self._train_basics()

        batch_size = data.batch_size
        units = c['model.units']
        self.alfa = c['model.alfa']

        self.X = tf.placeholder(tf.float32, shape=[batch_size, data.num_files,  data.timesteps, data.dim], name="X")
        self.tf_batch_size = tf.shape(self.X)[0]

        # add noise to input
        self.layer = self.X

        # deepfuck
        with tf.variable_scope('deepfuck'):
            self.layer = tf.layers.batch_normalization(self.layer, training=tf.equal(self.training, 1))

            # cells
            with tf.variable_scope('cell_fw'):
                cell_fw = tf.nn.rnn_cell.GRUCell(units[0])

            self.layer = tf.reshape(self.layer, [batch_size * data.num_files, data.timesteps, data.dim])  # [b, num_files, t, dim] => [b*num_files, t, dim]

            self.output, _ = tf.nn.dynamic_rnn(cell_fw, self.layer, dtype=tf.float32)  # => [b*num_files, t, units[0]]

            self.output = self.time_distributed_dense(self.output, [batch_size * data.num_files, data.timesteps, units[0]], units[1]) # => [b*num_files, t, units[1]]

            # For summary
            self.rnn_output = tf.transpose(tf.reshape(self.output, [batch_size * data.num_files, data.timesteps, units[1], 1])[:10], perm=[0, 2, 1, 3])

            self.rnn_summary = tf.summary.image('rnn_output', self.rnn_output)
            self.update_ops += [self.rnn_summary]

            self.layer = tf.reduce_mean(self.output, axis=1)  # => [b*num_files, units[1]]

            #self.layer = tf.layers.dense(self.layer, units[-1])

            self.triplets = tf.reshape(self.layer, [batch_size, data.num_files, units[1]])  # => [b, num_files, units[1]]

            #self.triplets = tf.tanh(2*tf.tanh(self.triplets))

            self.l2 = self.triplets / tf.linalg.norm(self.triplets, axis=-1, keepdims=True)  # => [b, num_files, units[1]]

            # For summary
            self.dvectors = tf.reshape(self.triplets, [batch_size, data.num_files, units[1], 1])[:10]

            self.dvector_summary = tf.summary.image('dvectors', self.dvectors)
            self.update_ops += [self.dvector_summary]

        out = tf.identity(self.l2, name="output")
        self.out = self.l2

        # cost & optimizer
        with tf.name_scope("cost_optimizer"):
            # loss function
            if c['model.loss'] == 'triplet_l2':
                pass

            elif c['model.loss'] == 'triplet_cos':
                self.positive_similarity = tf.einsum("ijk,ijk->ij", self.l2[:, 0:1], self.l2[:, 1:2])  # => [b, 1]
                self.negative_similarity = tf.einsum("ijk,ijk->ij", self.l2[:, 0:1], self.l2[:, 2:3])  # => [b, 1]
                self.loss = self.negative_similarity - self.positive_similarity + self.alfa  # => [b, 1]
                self.loss = tf.maximum(self.loss, 0)
                self.cost = tf.reduce_mean(self.loss, axis=0)[0]  # []
            elif c['model.loss'] == 'margin':
                pass

            else:
                raise Exception('Incorrect loss in config')

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