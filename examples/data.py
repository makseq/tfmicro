#!/usr/bin/env python
# noispection PyPep8Naming
import csv
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import tfmicro.data
import tfmicro.threadgen

# Data: bitstamp.net/api/transactions.csv


# noinspection PyUnresolvedReferences
class Data(tfmicro.data.Data):

    def __init__(self, c):
        self.c = c
        x, t = self.read(c['data.path'], c['data.period'],
                         c['data.price_step'], c['data.resolution_ratio'], c['data.cache'])

        # take data from start date only
        if isinstance(c['data.start_from'], str) or isinstance(c['data.start_from'], unicode):  # date format
            dt = np.datetime64(c['data.start_from'])
            start_sec = (dt - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')

            try:
                start = next(idx for idx, val in np.ndenumerate(t) if val == int(start_sec + 0.5))[0]
            except:
                print 'data.start_from is incorreect. Use it multiple by data.period and less end time'
                exit(-1)

        else:  # number
            start = c['data.start_from']

        print 'Start date of quotes:', np.datetime64(np.int64(t[start]), 's')
        x, t = x[start:], t[start:]

        # preparers
        x = self.roll(x)
        self.prepare(x, t)
        self.unit_tests()

    def roll(self, x):
        x = x[:, 0:4].toarray()  # take only N dim we need
        self.preroll_x = np.copy(x[:, 0:4])

        # log
        if self.c['data.log.before_diff']:
            x.data[:, 0:2] = np.log(x.data[:, 0:2])

        if self.c['data.diff']:
            x[1:, 0:2] -= x[:-1, 0:2]
            x[0, 0:2] = 0

        # log
        if self.c['data.log.after_diff']:
            x.data[:, 0:2] = np.log(x.data[:, 0:2])
        return x  # sp.csr_matrix(x)

    def unroll(self, d, start):
        is_ndarray = isinstance(d, np.ndarray)

        if self.c['data.log.after_diff']:
            d = np.array(d)
            d[:, 0:2] = np.exp(d[:, 0:2])

        # revert
        if self.c['data.diff']:
            x = np.zeros([d.shape[0] + 1, d.shape[1]])
            x[0] = self.preroll_x[start-1] if start is not None else 0
            if self.c['data.log.before_diff']:
                x[0, 0:2] = np.log(x[0, 0:2])
            x[1:] = d[:]
            x[:,  0:2] = np.cumsum(x[:, 0:2], axis=0)
            x = x[1:]
        else:
            x = np.copy(d)

        if self.c['data.log.before_diff']:
            x[:, 0:2] = np.exp(x[:, 0:2])

        return x if is_ndarray else sp.csr_matrix(x)

    def output_inverse_transform(self, x, local_mean, local_std):
        x = np.copy(x)

        # local normalization
        if self.c['data.norm.local']:
            x *= local_std
            x += local_mean

        # global normalization
        if self.c['data.norm.global']:
            x *= self.std
            x += self.mean
        return x

    def output_transform(self, x, local_mean, local_std, inplace=False):
        if not inplace:
            x = np.copy(x)

        # global normalization
        if self.c['data.norm.global']:
            x -= self.mean
            x /= self.std

        # local normalization
        if self.c['data.norm.local']:
            x -= local_mean
            x /= local_std

        return x

    def input_transform(self, x, inplace=False):
        if not inplace:
            x = np.copy(x)

        # global normalization
        if self.c['data.norm.global']:
            x -= self.mean
            x /= self.std

        # local normalization
        if self.c['data.norm.local']:
            local_mean = np.mean(x, axis=0)
            local_std = np.std(x, axis=0)
            #local_std[local_std < 1e-3] = local_mean[local_std < 1e-3] * 0.01

            x -= local_mean
            x /= local_std

            local_std *= self.c['data.norm.local.std_coef']
        else:
            local_mean, local_std = None, None

        return x, local_mean, local_std

    def prepare(s, x, t):
        c = s.c
        s.t, s.x = t, x

        s.input_len, s.output_len = c['data.input_len'], c['data.output_len']
        s.input_dim, s.output_dim = x.shape[1], x.shape[1]
        s.output_offset = c['data.output_offset']
        s.hop = s.c['data.hop']
        s.len = x.shape[0]
        s.batch_size = c['model.batch_size']

        train_ind, test_ind = s.split()
        s.steps_per_epoch = int(np.ceil(len(train_ind)/float(s.batch_size)))
        s.validation_steps = int(np.ceil(len(test_ind)/float(s.batch_size)))

        # take only train part
        x = x[0: test_ind[0]]

        # calculate global normalization
        tmp = x if isinstance(x, np.ndarray) else x.toarray()
        s.mean = np.mean(tmp, axis=0)
        std = np.std(tmp, axis=0)
        dim_zero = np.sum(std == 0)

        # exclude zero values from std
        prev = std[0]
        for i, v in enumerate(std):
            prev = v if v != 0 else prev
            std[i] = prev if v == 0 else std[i]
        s.std = std
        print ' global norm\n  std ', s.std, '\n  mean', s.mean

        s.dim_zero = dim_zero
        s.dim_nonzero = s.input_dim - dim_zero

        print ' start  date', np.datetime64(np.int64(t[0]), 's')
        print ' finish date', np.datetime64(np.int64(t[-1]), 's')
        print ' data total', s.len, 'timesamples', s.len / s.hop, 'examples', s.len / s.hop / s.batch_size, 'batches'
        print ' input len', s.input_len, 'input dim',  x.shape[1], 'output dim', x.shape[1] * s.output_len, \
              ' dim_nonzero', s.dim_nonzero

    def split(self):
        assert len(self.t) == self.x.shape[0]
        hop = self.c['data.hop']
        offset = self.output_offset
        begin = (len(self.t) - self.output_len - offset) % hop
        ind = np.arange(begin, len(self.t) - self.output_len - offset + 1, hop)
        train_part = int(len(ind) * self.c['data.train_part'] + 0.5)
        train = ind[0:train_part]
        test = ind[train_part:]
        return train, test

    @tfmicro.threadgen.threadsafe_generator
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

                def task(args):
                    i, k = args

                    # timing[k] = i / float(len(self.t))

                    inp = x[i: i + input_len]
                    inp = inp if isinstance(inp, np.ndarray) else inp.toarray()
                    try:
                        inputs[k] = inp
                    except:
                        print 'FIXME: BUG'

                    out = x[i + offset: i + offset + output_len]
                    out = out if isinstance(out, np.ndarray) else out.toarray()
                    outputs[k] = out

                    _, local_mean, local_std = self.input_transform(inputs[k], inplace=True)
                    self.output_transform(outputs[k], local_mean, local_std, inplace=True)

                pool.map(task, zip(w, range(len(w))))

                #for k, work_i in enumerate(w):
                #    task([work_i, k])

                for i in range(len(outputs)):
                    if np.any(outputs[i] < -1e+7):
                        print i
                        print
                        print inputs[i]
                        print
                        print outputs[i]

                yield inputs, outputs

    @staticmethod
    def draw_hist(hist, bins):
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()

    @staticmethod
    def plot(x, o=None):
        fig, ax = plt.subplots()

        # x
        ax.plot(x[:, 0], color='b')
        ax.plot(x[:, 1], color='g')

        # original
        if o is not None:
            ax.plot(o[:, 0], color='r')
            ax.plot(o[:, 1], color='y')

        plt.show()

    @staticmethod
    def read(path, period, resolution_price_step, resolution_ratio, cache_dir):
        # make cache file name
        name = os.path.splitext(os.path.basename(path))[0]
        cache_path = os.path.join(cache_dir, name) \
                     + '_' + str(period) \
                     + '_step' + str(resolution_price_step) \
                     + '_ratio' + str(resolution_ratio)
        random = np.random.RandomState(42)

        # load cache from files
        if os.path.exists(cache_path + '.samples.npz') and cache_dir:
            print 'try to load cache:', cache_path
            t = np.load(cache_path + '.time.npy')
            x = sp.load_npz(cache_path + '.samples.npz')
            return x, t

        # read from file
        print 'loading data:', path
        f = open(path, 'rb')
        reader = csv.reader(f, delimiter=',')
        next(reader, None)  # skip header
        c, data, time = 0, [], []
        for row in reader:
            idn, timestamp, amount, price = row
            c += 1
            time += [int(timestamp)]
            amount = np.float32(amount)
            price = np.float32(price)
            data += [np.array([amount, price], dtype=np.float64)]

        # prepare
        time = np.array(time, dtype=np.int32)
        data = np.array(data, dtype=np.float64)
        ind = np.argsort(time)  # time sort
        time = time[ind]
        data = data[ind]

        start = time[0] - time[0] % period  # start from number multiple period
        end = time[-1]

        print 'start  date:', np.datetime64(np.int64(start), 's')
        print 'finish date:', np.datetime64(np.int64(end), 's')
        print 'period:', period

        # find max and min
        time_range = xrange(start, end, period)
        range_end = range_start = 0
        d_min, d_max = [], []
        for i in time_range:
            while range_end < len(time) and time[range_end] < i + period:
                range_end += 1
            if range_end - range_start > 0:
                d = data[range_start:range_end]
                d_min += [np.min(d[:, 1])]
                d_max += [np.max(d[:, 1])]
            range_start = range_end
            if i % 10 == 0:
                print '\r find max diff %0.4f' % (float(i - start) / float(end - start)),
        print

        # find max and min diff price
        d_min, d_max = np.array(d_min), np.array(d_max)
        max_price = np.max(d_max - d_min)
        min_price = np.min(d_max - d_min)
        print 'diff max_price', max_price, 'min_price', min_price
        print 'ratio', resolution_ratio, 'total', np.ceil(max_price * resolution_ratio), 'step', resolution_price_step

        # parameters setup
        t, o = [], 5

        max_price = np.ceil(max_price) * resolution_ratio
        even = max_price + (resolution_price_step - max_price % resolution_price_step)
        bins = np.arange(0, even+resolution_price_step, resolution_price_step)
        bins_number = len(bins) - 1

        prev_price = np.zeros([o-1])
        range_end = range_start = 0
        print 'feature dim', o + 2 * bins_number
        # out = np.zeros((len(time_range), o + 2 * bins_number), dtype=np.float64)
        x = sp.lil_matrix((len(time_range), o + 2 * bins_number), dtype=np.float64)

        # histogram resampling
        c = 0
        for i in time_range:
            while range_end < len(time) and time[range_end] < i + period:
                range_end += 1

            if range_end - range_start > 0:
                d = data[range_start:range_end]
                d[d == 0] = np.finfo(np.float32).eps
                min_price, max_price = np.min(d[:, 1]), np.max(d[:, 1])

                prices = d[:, 1] - min_price
                prices[prices < 0] = 0
                h_volumes = np.histogram(prices, bins=bins, weights=d[:, 0])[0]
                h_numbers = np.histogram(prices, bins=bins)[0]

                x[c, 0] = min_price
                x[c, 1] = max_price + 1e-7
                x[c, 2] = np.sum(h_volumes)
                x[c, 3] = np.sum(h_numbers)
                x[c, 4] = i - start

                x[c, o: o + bins_number] = h_volumes / x[c, 2]
                x[c, o + bins_number: o + 2 * bins_number] = h_numbers / x[c, 3]

                prev_price = x[c, 0:4].toarray()[0]
            else:
                x[c, 0] = prev_price[0]  # - 1e-7 * random.rand()
                x[c, 1] = prev_price[1]  # + 1e-7 * random.rand()
                x[c, 2] = prev_price[2]
                x[c, 3] = prev_price[3]
                x[c, 4] = i - start

            if i % 10 == 0:
                print '\r sampling %0.4f' % (float(i - start) / float(end - start)),
            range_start = range_end
            t += [i]
            c += 1
        print

        t = np.array(t, dtype=np.int32)
        x = x.tocsr()

        # save cache to files
        if cache_dir:
            print 'saving cache:', cache_path
            os.makedirs(cache_dir) if not os.path.exists(cache_dir) else ()
            np.save(cache_path + '.time.npy', t)
            sp.save_npz(cache_path + '.samples.npz', x)
            # np.save(cache_path + '.samples.npy', out)
        return x, t

    def unit_tests(self, debug=False):
        s = self
        print 'Unit testing'

        # Test 0: data consistent
        if not self.c['data.log.before_diff'] and not self.c['data.log.after_diff']:
            assert not np.any(self.preroll_x[:, 0] > self.preroll_x[:, 1])  # min > max check, 'Test 0: min > max'

        # Test 1: transform -> inverse_transform -> unroll == should be equal to preroll
        input_len = self.c['data.input_len']
        output_len = self.c['data.output_len']
        raw = self.x[-output_len-input_len:]
        raw = raw if isinstance(raw, np.ndarray) else raw.toarray()
        inp, mean, std = s.input_transform(raw)
        if debug: print 'local mean', mean, 'std', std

        x1 = self.x[-output_len:]
        x1 = x1 if isinstance(x1, np.ndarray) else x1.toarray()
        x2 = s.output_transform(x1, mean, std)
        x3 = s.output_inverse_transform(x2, mean, std)
        if debug: self.plot(x3)
        x4 = s.unroll(x3, -output_len)
        if debug: self.plot(x4)

        out = s.preroll_x[-output_len:]
        test1 = np.sum(np.abs(x4 - out))
        if debug: print 'Test 1: sum', test1
        assert test1 < 1e-8, 'Test 1 failed: transform -> inverse_transform -> unroll == preroll: ' + str(test1)

if __name__ == '__main__':
    import json
    d = Data(json.load(open('config/config_predict.json')))

    train, test = d.split()
    print test[-1], d.x.shape[0], d.x.shape[0]-test[-1]
    exit()

    t, x = d.t, d.x
    x = x.tocsr()
    arg_max = x[:, 0].argmax()
    print 'max price at', np.datetime64(np.int64(t[arg_max]), 's'), '\t\t\n', x[arg_max]

    d.unit_tests()

    plot_x, _, _ = d.input_transform(x[-1000:].toarray())
    print np.mean(plot_x, axis=0)
    print np.std(plot_x, axis=0)
    d.plot(plot_x, plot_x[:, 2:])

    # all
    # d.plot(d.preroll_x, d.preroll_x)



