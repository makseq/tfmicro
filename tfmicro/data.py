#!/usr/bin/env python
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
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import threadgen


# noinspection PyPep8Naming
# noinspection PyUnresolvedReferences
class Data(object):
    def __init__(self, c):
        self.c = c
        self.prepare(x, t)
        self.unit_tests()

    # some input transform (e.g.: z-norm)
    def input_transform(self, x, inplace=False):
        return None

    # prepare your data here
    def prepare(s, x, t):
        c = s.c

    # split data into train & validation sets
    def split(self):
        return train, test

    # generator
    @threadgen.threadsafe_generator
    def generator(self, mode):
        c, t, x = self.c, self.t, self.x
        input_len, output_len = self.input_len, self.output_len
        offset = self.output_offset
        train_ind, test_ind = self.split()
        random = np.random.RandomState(42)
        pool = multiprocessing.pool.ThreadPool(4)
        # print ' Generator', mode, 'created'

        # make work
        train1 = np.copy(train_ind)
        random.shuffle(train1)
        work = {'train': train_ind, 'valid': test_ind, 'test': test_ind,
                'train1': train1[-self.batch_size:]}[mode]
        while 1:
            work = np.copy(work)
            # print '\n\n' + mode + str(len(work)) + '/' + str(self.batch_size) + '\n\n'

            if c['data.shuffle']:
                random.shuffle(work)

            for j in xrange(0, len(work), self.batch_size):
                w = work[j: j + self.batch_size]
                # timing = np.zeros((len(w), 1))
                inputs = np.zeros((len(w), input_len, x.shape[1]))
                outputs = np.zeros((len(w), output_len, x.shape[1]))

                # multi threading task
                def task(args):
                    i, k = args
                    inp = x[i: i + input_len]
                    inp = inp if isinstance(inp, np.ndarray) else inp.toarray()
                    inputs[k] = inp

                    out = x[i + offset: i + offset + output_len]
                    out = out if isinstance(out, np.ndarray) else out.toarray()
                    outputs[k] = out

                    _, local_mean, local_std = self.input_transform(inputs[k], inplace=True)
                    self.output_transform(outputs[k], local_mean, local_std, inplace=True)

                # pool
                pool.map(task, zip(w, range(len(w))))

                # no multi threading
                # for k, work_i in enumerate(w):
                #    task([work_i, k])

                yield inputs, outputs

    # unit testing
    def unit_tests(self, debug=False):
        # some tests here
        raise Exception('No unit tests!')


if __name__ == '__main__':
    import json
    d = Data(json.load(open('config/config.json')))

    # some tests here






