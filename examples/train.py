#!/usr/bin/env python

'''
 This is template only, but NOT WORKING example. 
'''

# cuda export: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64
# -----------------------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['PYTHONHASHSEED'] = '0'

# matplotlib use Agg for headless systems
if os.path.exists('mpl.agg'):
    import matplotlib
    matplotlib.use('Agg')
    mpl_agg = True
    print "Note: matplotlib.use('Agg') enabled, no plots will be available!"
else:
    mpl_agg = False

import numpy as np
import random
np.random.seed(42)
random.seed(12345)
import tensorflow as tf
tf.set_random_seed(1234)

# Local imports
import testarium
from tfmicro import callbacks, callback_plot

import my_model
import web
import predict
from data import Data


# get all callbacks you need
def build_callbacks(c, model_dir):
    cbs = []

    # Callback: Save the best model
    cbs += [callbacks.ModelCheckpoint(model_dir + '-valid', monitor='val_loss', save_best_only=True, start_from_epoch=10, max_models_to_keep=3)]
    cbs += [callbacks.ModelCheckpoint(model_dir + '-train', monitor='loss', save_best_only=True, start_from_epoch=5)]
    cbs += [callbacks.KeyboardLearningRate()]
    cbs += [callbacks.KeyboardStop()]
    cbs += [callbacks.ReducingLearningRate()]

    if c['callbacks.plot']:
        cbs += [callback_plot.PlotCallback()]
    return cbs


# Train
@testarium.experiment.set_run
def run(commit):
    print('Init')
    c = commit.config
    # disable plots for headless systems
    if mpl_agg:
        c['callbacks.plot'] = False
        c['debug.plot'] = False

    # paths
    model_dir = '../last/model'

    # init & data loading
    data = Data(c)
    commit.model = my_model.MyModel(c)

    # TRAIN
    if not c['model.skip_train']:

        # load model if we want continue training
        if 'model.preload' in c and c['model.preload']:
            print 'FIXME: Model preloaded from', c['model.preload']
            exit(-1)

        # train
        print 'Training'
        commit.MakeLink()
        callbacks_list = build_callbacks(c, model_dir)
        commit.model.fit_data(data, max_queue_size=10, callbacks=callbacks_list, epochs=c['model.epochs'], use_gpu=c['use_gpu'],
                              tensorboard_subdir=commit.name)
        commit.model.save(model_dir + '-last')

    # TEST
    #predict.test1(commit, data)
    return 0


# Scoring
@testarium.experiment.set_score
def score(commit):
    h = commit.model.history
    loss, val_loss = h['loss'][-1], h['val_loss'][-1]
    commit.MakeGraph('plot_loss.json', h['loss'], 'epoch', 'loss')  # plots for testarium
    commit.MakeGraph('plot_valid.json', h['val_loss'], 'epoch', 'validation loss')
    return {'score': loss, 'valid': val_loss, 'history': h}


if __name__ == '__main__':
    testarium.testarium.best_score_is_min()
    testarium.main()
