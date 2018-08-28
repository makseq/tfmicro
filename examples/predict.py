#!/usr/bin/env python

'''
 This is template only, but NOT WORKING example. 
'''

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
    predicted = p.predict(inp)
    return predicted


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

if __name__ == '__main__':
    pass
