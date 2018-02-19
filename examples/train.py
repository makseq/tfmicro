#!/usr/bin/env python
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

import numpy as np, random
np.random.seed(42)
import tensorflow as tf
random.seed(12345)
tf.set_random_seed(1234)

# Local imports
import testarium
import my_model, predict
from tfmicro import callbacks, callback_plot
from data import Data


# get all callbacks you need
def build_callbacks(c, model_dir):
    cbs = []

    # Callback: Save the best model
    cbs += [
        callbacks.ModelCheckpoint(model_dir + '-cross', monitor='val_loss', save_best_only=True, start_from_epoch=10, max_models_to_keep=3)]
    cbs += [callbacks.ModelCheckpoint(model_dir + '-train', monitor='loss', save_best_only=True, start_from_epoch=5)]
    cbs += [callbacks.KeyboardLearningRate()]
    cbs += [callbacks.KeyboardStop()]

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
        commit.model.fit_data(data, callbacks=callbacks_list, epochs=c['model.epochs'], use_gpu=c['use_gpu'],
                              tensorboard_subdir=commit.name)
        commit.model.save(model_dir + '-last')

    # TEST
    predict.test1(commit, data)
    return 0


# Scoring
@testarium.experiment.set_score
def score(commit):

    h = commit.model.history
    loss, val_loss = h['loss'][-1], h['val_loss'][-1]
    commit.MakeGraph('plot_loss.json', h['loss'], 'epoch', 'loss')  # plots for testarium
    commit.MakeGraph('plot_cross.json', h['val_loss'], 'epoch', 'cross validation loss')
    #except:
    #    h, loss, val_loss = 'None', 100, 100
    return {'score': loss, 'cross': val_loss, 'history': h}


# Print
@testarium.testarium.set_print
def print_console(commit):
    return ['name', 'loss', 'cross', 'time', 'comment', 'loss_plot', 'cross_plot'], \
           [commit.name,
            '%0.3f' % (commit.desc['score']),
            '%0.3f' % (commit.desc['cross']),
            '%0.0f' % (commit.desc['duration']),
            str(commit.desc['comment']).replace('{','').replace('}','').replace('"','').replace('[','').replace(']',''),
            'graph://storage/' + commit.dir + '/plot_loss.json',
            'graph://storage/' + commit.dir + '/plot_cross.json',
            ]


if __name__ == '__main__':
    testarium.testarium.best_score_is_min()
    testarium.main()
