import tensorflow as tf
from tfmicro import model
import my_cell


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
    def time_distributed_dense(layer, units, output_dim, weights, bias):
        shape = tf.shape(layer)
        layer = tf.reshape(layer, [shape[0] * shape[1], shape[2]])
        layer = tf.matmul(layer, weights) + bias  # [m, units] x [n, output_dim] = [m, output_dim]
        layer = tf.reshape(layer, [shape[0], shape[1], output_dim])  # data.output_dim
        return layer

    # loop function
    def loop_nest(self, cell):
        data = self.data
        c = self.c

        # ATTENTION
        T = 0  # data.output_len
        A = self.encoder_units[-1]  # data.input_dim
        M = self.encoder_units[-1]  # data.input_dim  # self.encoder_units[-1]
        P = self.decoder_units[0]

        with tf.variable_scope('attention'):
            UP = 128
            init = tf.glorot_uniform_initializer()
            self.U = tf.Variable(init([T+1, A, UP]))  # A
            self.W = tf.Variable(init([T+1, P, UP]))
            self.V = tf.Variable(init([T+1, 2*UP, M]))  # W concat U => M*2
            self.bias_W = tf.Variable(tf.truncated_normal(shape=[T+1, UP], stddev=0.1))
            self.bias_U = tf.Variable(tf.truncated_normal(shape=[T+1, UP], stddev=0.1))
            self.bias_V = tf.Variable(tf.truncated_normal(shape=[T+1, M], stddev=0.1))

            '''# cells
            with tf.variable_scope('cell_fw'):
                cell_fw = tf.nn.rnn_cell.LSTMCell(A/2)
                if c['model.attention.keep_prob'] < 1:
                    cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=self.encoder_keep_prob)
            with tf.variable_scope('cell_bw'):
                cell_bw = tf.nn.rnn_cell.LSTMCell(A/2)
                if c['model.attention.keep_prob'] < 1:
                    cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=self.encoder_keep_prob)

            bidirectional dynamic rnn without state (stateless) for attention coefs
            rnn_attention, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.output_encoder, dtype=tf.float32)
            self.rnn_attention = tf.concat(rnn_attention, axis=-1)'''

            self.rnn_attention = self.output_encoder
            self.att_input = self.output_encoder
            attention_scope = tf.get_variable_scope()

            '''dim = M
            self.attention_mat = tf.Variable(tf.random_normal([data.output_len + 1, data.input_len, dim], stddev=0.5))
            a_input = tf.reshape(self.output_encoder, [-1, 1, data.input_len, dim], name='reshape_input')
            # [batch, 1, input_len, dim] * [output_len, input_len, dim]
            a_out = a_input * self.attention_mat  # => [batch, output_len, input_dim, dim]
            attention = tf.reduce_sum(a_out, axis=2)  # sum by input_dim => [batch, output_len, dim]
            print '  attention shape', attention.get_shape()'''
            # self.att_context = attention[:, t, :]

        def loop_fn(t, cell_output, cell_state, loop_state):
            next_loop_state, emit_output = loop_state, cell_output
            elements_finished = t >= data.output_len
            next_input = cell_output
            next_cell_state = cell_state

            # INIT
            if cell_output is None:  # time == 0
                next_cell_state = cell.zero_state(self.tf_batch_size, tf.float32)  #self.state_encoder
                next_input = cell.zero_state(self.tf_batch_size, tf.float32)[-1].h  # self.output_encoder[:, -1, :]

            # ATTENTION
            with tf.variable_scope(attention_scope, reuse=tf.AUTO_REUSE):
                oldt = t
                t = 0
                batch_size = self.tf_batch_size
                print '   next input', next_input.get_shape()

                # Uh: encoder lstm
                Uh = tf.einsum('bia,au->biu', self.rnn_attention, self.U[t]) + self.bias_U[t]  # [batch, i, A] x [A, UP] = [batch, i, UP]
                Uh = tf.transpose(Uh, [1, 0, 2])  # [batch, i, UP] => [i, batch, UP]
                print '   Uh', Uh.get_shape()

                # Wd: decoder lstm
                Wd = tf.matmul(next_input, self.W[t]) + self.bias_W[t]  # [batch, P] x [P, UP] = [batch, UP]
                Wd = tf.reshape(Wd, [1, batch_size, UP])
                Wd = tf.tile(Wd, [data.input_len, 1, 1])
                print '   Wd', Wd.get_shape()

                kernel = tf.concat([Uh, Wd], axis=-1)  # [i, batch, UP] ; [i, batch, UP] = [i, batch, 2*UP]
                print '   ker', kernel.get_shape()
                # kernel = tf.layers.batch_normalization(kernel, training=tf.equal(self.training, 1), reuse=tf.AUTO_REUSE)
                kernel = tf.nn.tanh(kernel)

                l = tf.einsum('ibu,um->ibm', kernel, self.V[t]) + self.bias_V[t]  # [i, batch, 2*UP] x [2*UP, M] + [M] => [i, batch, M]
                print '   l', l.get_shape()

                b = tf.exp(l)  # [i, batch, M]
                b = b / tf.reduce_sum(b, axis=0)  # [i, batch, M] / [batch, sum(M_i)]
                b = tf.transpose(b, [1, 0, 2])  # => [batch, i, M]

                '''b = tf.Print(b, [b[0, :, 0]], summarize=20, message='b ')
                b = tf.Print(b, [self.V[0]], summarize=100, message='V ===')
                b = tf.Print(b, [self.W[0]], summarize=100, message='W ===')
                b = tf.Print(b, [self.U[0]], summarize=100, message='U ===')'''

                self.att_context = tf.reduce_sum(b * self.att_input, axis=1)
                print '   att_context', self.att_context.get_shape()

            next_input = tf.concat([next_input, self.att_context], axis=-1)
            print '   cell output', next_input.get_shape()

            print
            # return
            return elements_finished, next_input, next_cell_state, emit_output, next_loop_state
        return loop_fn

    def _train_model(self, data):
        c = self.c
        self._train_basics()

        batch_size = data.batch_size if c['model.stateful'] else None
        self.encoder_units = encoder_units = c['model.encoder.units']
        self.decoder_units = decoder_units = c['model.decoder.units']

        self.X = tf.placeholder(tf.float32, shape=[batch_size, data.input_len,  data.input_dim], name="X")
        self.Y = tf.placeholder(tf.float32, shape=[batch_size, data.output_len, data.output_dim], name="Y")
        self.encoder_keep_prob = tf.placeholder_with_default(1.0, shape=(), name='encoder_keep_prob')
        self.decoder_keep_prob = tf.placeholder_with_default(1.0, shape=(), name='decoder_keep_prob')
        self.tf_batch_size = tf.shape(self.X)[0]

        # add noise to input
        layer = self.X
        if self.c['model.input_noise'] > 0:
            noise = tf.random_normal(shape=tf.shape(layer), dtype=tf.float32, stddev=self.c['model.input_noise'])
            layer = tf.cond(tf.equal(self.training, 1),
                            true_fn=lambda: layer + noise,
                            false_fn=lambda: layer)

        # ENCODER
        with tf.variable_scope('encoder'):
            # cells
            with tf.variable_scope('cell_fw'):
                cell_encoder_fw = tf.nn.rnn_cell.LSTMCell(encoder_units[0]/2)
                if c['model.encoder.keep_prob'] < 1:
                    cell_encoder_fw = tf.nn.rnn_cell.DropoutWrapper(cell_encoder_fw,
                                                                    input_keep_prob=self.encoder_keep_prob)
            with tf.variable_scope('cell_bw'):
                cell_encoder_bw = tf.nn.rnn_cell.LSTMCell(encoder_units[0]/2)
                if c['model.encoder.keep_prob'] < 1:
                    cell_encoder_bw = tf.nn.rnn_cell.DropoutWrapper(cell_encoder_bw,
                                                                    input_keep_prob=self.encoder_keep_prob)

            # bidirectional dynamic rnn without state (stateless)
            self.output_encoder, _ = tf.nn.bidirectional_dynamic_rnn(cell_encoder_fw, cell_encoder_bw, layer, dtype=tf.float32)
            self.output_encoder = tf.concat(self.output_encoder, 2)

            next_units = encoder_units[1:]
            prev_input = layer
            for i in xrange(len(next_units)):
                with tf.variable_scope('encoder_' + str(i)):
                    cell = tf.nn.rnn_cell.LSTMCell(next_units[i])
                    if c['model.encoder.keep_prob'] < 1:
                        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.encoder_keep_prob)

                    input_encoder = tf.concat([self.output_encoder, prev_input], -1) if i < 2 \
                        else tf.add(self.output_encoder, prev_input)

                    prev_input = self.output_encoder
                    self.output_encoder, self.state_encoder = tf.nn.dynamic_rnn(cell, input_encoder, dtype=tf.float32)

            print '  encoder output:', self.output_encoder.get_shape()

        # DECODER
        with tf.variable_scope('decoder'):
            # Cells
            decoder_cells = []
            for i in xrange(len(decoder_units)):
                last = i == len(decoder_units) - 1
                with tf.variable_scope('decoder_' + str(i)):
                    # main cell
                    cell_decoder = tf.nn.rnn_cell.LSTMCell(decoder_units[i])

                    # drop out
                    '''if c['model.decoder.keep_prob'] < 1:
                        cell_decoder = tf.nn.rnn_cell.DropoutWrapper(cell_decoder, output_keep_prob=self.decoder_keep_prob)'''

                    # attention & residual
                    cell_decoder = my_cell.MyCellWrapper(cell_decoder, last)

                    decoder_cells += [cell_decoder]
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cells)  # tf.contrib.rnn.MultiRNNCell(decoder_cells)

            # raw rnn
            outputs_ta, _, _ = tf.nn.raw_rnn(cell=multi_cell, loop_fn=self.loop_nest(multi_cell))
            outputs = outputs_ta.stack()
            layer = tf.transpose(outputs, [1, 0, 2])

        # Output matmul
        output_dim = 2  # data.output_dim  # FIXME: make 4 dim!
        bias = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        weights = tf.Variable(tf.truncated_normal([decoder_units[-1], output_dim], stddev=0.5))
        layer = self.time_distributed_dense(layer, decoder_units[-1], output_dim, weights, bias)

        out = tf.identity(layer, name="output")
        self.out = out

        # cost & optimizer
        with tf.name_scope("cost_optimizer"):
            # loss function
            if c['model.loss'] == 'cumsum-mse':
                diff = tf.reduce_mean(tf.square(tf.cumsum(self.Y[:, :, 0:2], axis=1) - tf.cumsum(out[:, :, 0:2], axis=1)))
            elif c['model.loss'] == 'mse':
                diff = tf.reduce_mean(tf.square(self.Y[:, :, 0:2] - out[:, :, 0:2]))
            elif c['model.loss'] == 'minmax':
                diff = self.minmax_loss()
            elif c['model.loss'] == 'minmax+cumsum':
                cumsum = tf.reduce_mean(tf.square(tf.cumsum(self.Y[:, :, 0:2], axis=1) - tf.cumsum(out[:, :, 0:2], axis=1)))
                diff = self.minmax_loss() * 0.25 + cumsum * 0.75
            elif c['model.loss'] == 'lazy-mse':
                diff = self.lazy_mse()

            else:
                raise Exception('Incorrect loss in config')

            self.cost = tf.clip_by_value(diff, 1e-40, 1e10)
            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_tf)
            # gradient clip
            gradients, variables = zip(*self.optimizer.compute_gradients(self.cost))
            gradients = [None if gradient is None else tf.clip_by_norm(gradient, 1e+10) for gradient in gradients]
            self.optimizer = self.optimizer.apply_gradients(zip(gradients, variables))

    def lazy_mse(self):
        Y = tf.cumsum(self.Y, axis=1)
        out = tf.cumsum(self.out, axis=1)
        tf_batch_size = tf.shape(self.X)[0]

        c = 4
        summa = tf.zeros([tf_batch_size, 2])
        for i in xrange(0, self.data.output_len-c/2, c/2):
            y = tf.reduce_mean(Y[:, i:i+c, 0:2], axis=1)
            o = tf.reduce_mean(out[:, i:i+c, 0:2], axis=1)
            summa = summa + tf.square(y - o, name='minusss')
        summa = summa / (float(self.data.output_len) / float(c))
        summa = tf.reduce_mean(summa)

        '''i = tf.constant(0)
        c = lambda i: tf.less(i, 10)
        b = lambda i: tf.add(i, 1)
        r = tf.while_loop(c, b, [i])'''
        return summa

    def minmax_loss(self):
        Y = tf.cumsum(self.Y[:, :, 0:2], axis=1)
        out = tf.cumsum(self.out[:, :, 0:2], axis=1)

        min_real = tf.reduce_min(Y[:, :, 0:2], axis=1)
        max_real = tf.reduce_max(Y[:, :, 0:2], axis=1)
        min_pred = tf.reduce_min(out[:, :, 0:2], axis=1)
        max_pred = tf.reduce_max(out[:, :, 0:2], axis=1)

        diff1 = tf.reduce_mean(tf.square(min_pred - min_real)) + tf.reduce_mean(tf.square(max_pred - max_real))
        return diff1/2.0

    def train_step(self):
        c = self.c
        self.x, self.y = self.train_generator.get_values()

        # train step
        params = [self.cost, self.summary, self.optimizer, self.out] + self.update_ops
        out = self.sess.run(params, feed_dict={
            self.X: self.x,
            self.Y: self.y,
            self.encoder_keep_prob: c['model.encoder.keep_prob'],
            self.decoder_keep_prob: c['model.decoder.keep_prob'],

            self.training: 1,
            self.step_tf: self.step,
            self.epoch_tf: self.epoch,
            self.learning_rate_tf: self.learning_rate
        })
        cost, summary, _, self.train_prediction = out[0:-len(self.update_ops)] if self.update_ops else out

        # print context

        self.train_writer.add_summary(summary, global_step=self.epoch)
        self.train_costs += [cost]

    def validation_step(self):
        self.test_x, self.test_y = self.valid_generator.get_values()

        params = [self.cost_summary, self.cost, self.out] + self.update_ops
        out = self.sess.run(params, feed_dict={self.X: self.test_x,
                                               self.Y: self.test_y})
        cost_summary, cost, self.test_prediction = out[0:-len(self.update_ops)] if self.update_ops else out

        self.test_costs += [cost]
        self.test_writer.add_summary(cost_summary, global_step=self.epoch)