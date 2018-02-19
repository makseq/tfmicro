#!/usr/bin/env python
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

import json
import os

import numpy as np

import my_model
from data import Data


class Predictor:
    def __init__(self, model_dir, data):
        self.data = data
        self.model_dir = model_dir
        self.c = None
        self.load_model()

    def load_model(self):
        print 'Loading model', self.model_dir
        if not os.path.exists(self.model_dir):
            print '! Error: model is not existing', self.model_dir
            raise Exception('model is not existing:' + self.model)
        self.model = my_model.MyModel.load(self.model_dir)
        self.c = self.model.c

    def predict(self, inp):
        inp = inp[np.newaxis]
        c, data, model = self.c, self.data, self.model
        # batch_size = data.batch_size if c['model.stateful'] else None
        predicted = model.predict(inp)[0]
        return predicted


def test_predict(i, data, p, out_len):
    inp = data.x[i: i + data.input_len]
    inp = inp if isinstance(inp, np.ndarray) else inp.toarray()
    out = data.x[i + data.output_offset: i + data.output_offset + out_len]
    out = out if isinstance(out, np.ndarray) else out.toarray()

    inp, local_mean, local_std = data.input_transform(inp)
    out = data.output_transform(out, local_mean, local_std)

    predicted = p.predict(inp)
    p_tmp = np.zeros([predicted.shape[0], 4])
    p_tmp[:, 0:2] = predicted
    predicted = p_tmp

    # adv calculations
    predicted = data.output_inverse_transform(predicted, local_mean, local_std)
    out = data.output_inverse_transform(out, local_mean, local_std)
    predicted = data.unroll(predicted, start=i+data.output_offset)
    out = data.unroll(out, start=i+data.output_offset)
    return predicted, out


def keyboard_spy(actions):
    from tfmicro.tfmicro import keyboard
    keyboard.reset()

    def stop_plot():
        actions['stop_plot'] = True
        print "User pressed 'q': stopping plot"

    # Collect events until released
    keyboard.listen_key('q', stop_plot)
    keyboard.start()


# Test 1: sequence to sequence
def test1(commit, data):
    # get data
    c = commit.config
    train_ind, test_ind = data.split()
    out_len = data.output_len
    work = c['predict.data_set'].upper()

    # Recover states
    print
    print 'Predicting', work
    model_dir = c['predict.model']
    p = Predictor(model_dir, data)

    actions = {'stop_plot': False}
    keyboard_spy(actions)

    # Make test
    mses = []
    for i in (train_ind if work == 'TRAIN' else test_ind):
        print ' test position', i
        predicted, out = test_predict(i, data, p, out_len)
        mse = np.mean(np.square(predicted[-data.output_len:, 0:2] - out[:, 0:2]))
        mses += [mse]
        print '  mse', mse

        # DEBUG
        if c['predict.plot'] and not actions['stop_plot']:
            data.plot(predicted[-data.output_len:], out)
            # data.plot(predicted[:, 2:], out[:, 2:])

    overall_mse = np.mean(mses)
    print ' overall mse:', overall_mse
    return overall_mse


# Make real prediction
def real_prediction(config):
    # init
    print 'Init'
    c = json.load(open(config))
    data = Data(c)
    p = Predictor(c['predict.model'], data)

    # get and roll input
    print 'Roll input'
    inp = data.x[-data.input_len: ]
    inp = inp if isinstance(inp, np.ndarray) else inp.toarray()
    inp, local_mean, local_std = data.input_transform(inp)

    # predict
    print 'Predicting'
    predicted = p.predict(inp)
    p_tmp = np.zeros([predicted.shape[0], 4])
    p_tmp[:, 0:2] = predicted
    predicted = p_tmp

    # unroll output
    print 'Unroll output'
    predicted = data.output_inverse_transform(predicted, local_mean, local_std)
    predicted = data.unroll(predicted, start=-data.input_len + data.output_offset)


    ''' First
    input time start 2018-01-24T03:00:00
    output time start 2018-01-31T15:00:00
    output time end 2018-02-01T03:00:00
    '''
    print ' input time start', np.datetime64(np.int64(data.t[-data.input_len]), 's')
    print ' output time start', np.datetime64(np.int64(data.t[-1] + c['data.period']), 's')
    print ' output time end', np.datetime64(np.int64(data.t[-1] + c['data.period']
                                                     + data.output_len * c['data.period']), 's')

    import matplotlib.pyplot as plt, mpld3
    import datetime

    fig, ax = plt.subplots()

    x = np.array([np.datetime64(np.int64(data.t[-1] + c['data.period'] + i * c['data.period']), 's').astype(datetime.datetime) for i in xrange(data.output_len)])
    ax.plot(x, predicted[:, 0:2])
    ax.grid(color='lightgray', alpha=0.5)
    #plt.show()
    name = 'out'
    mpld3.save_json(fig, open('../service/static/predictions/%s.json' % name, 'w'))


if __name__ == '__main__':
    import sys
    real_prediction(sys.argv[1])
