"""
TFMicro
Copyright (C) 2018 Maxim Tkachenko, Alexander Yamshinin

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

import sys
import traceback
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import keyboard

stop_training = False
stop_training_now = False


class Callback(object):
    def __init__(self):
        self.model = None
        self.config = None

    def on_start(self): pass

    def on_finish(self): pass

    def on_step_begin(self): pass

    def on_step_end(self): pass

    def on_validation_begin(self): pass

    def on_validation_step_begin(self): pass

    def on_validation_step_end(self): pass

    def on_validation_end(self): pass

    def on_epoch_begin(self): pass

    def on_epoch_end(self): pass

    def set_model(self, model):
        self.model = model

    def set_config(self, config):
        self.config = config
        self.c = config


# Model save at checkpoint
class ModelCheckpoint(Callback):
    def __init__(self, path, monitor='val_loss', save_best_only=True, start_from_epoch=0, max_models_to_keep=1):
        self.path, self.monitor, self.save_best_only = path, monitor, save_best_only
        self.min = None
        self.start_from_epoch = start_from_epoch
        self.saver = None
        self.max_to_keep = max_models_to_keep
        super(Callback, self).__init__()

    def on_start(self):
        var_list = self.model.saver._var_list
        self.saver = tf.train.Saver(var_list, max_to_keep=self.max_to_keep)

    def on_epoch_end(self):
        val = self.model.history[self.monitor][-1]
        if self.save_best_only:
            if (self.min is None or self.min > val) and self.model.epoch >= self.start_from_epoch:
                self.min = val
                self.model.save(self.path, saver=self.saver)
                sys.stdout.write(' > save:' + self.path)
        else:
            self.model.save(self.path, saver=self.saver)
            sys.stdout.write(' > save:' + self.path)


# Keyboard stop
class KeyboardStop(Callback):
    def __init__(self):
        global stop_training
        stop_training = False

        # stop
        def on_stop():
            global stop_training, stop_training_now
            # twice press
            if stop_training:
                stop_training_now = True
                self.model.info('\n  -> Key \'q\' pressed twice. Training will stop at the end of step! \n\n')
                return False  # stop listener
            # first press
            else:
                stop_training = True
                self.model.info('\n  -> Training will stop at the end of epoch! \n\n')
                return True  # continue listener

        # stop
        def on_resume():
            global stop_training, stop_training_now
            if stop_training_now:
                print "\n -> Sorry, can't resume train, double 'q' pressed"
            else:
                stop_training = False
                self.model.info('\n  -> Training will resume! \n\n')
                return True  # continue listener

        # Collect events until released
        keyboard.listen_key('q', on_stop)
        keyboard.listen_key('Q', on_resume)
        keyboard.start()

        print "! Note: press 'q' to stop training"
        print "! Note: press 'Q' to resume training"
        print "! Note: press 'q' twice to stop training after the step"
        super(Callback, self).__init__()

    def on_step_end(self):
        global stop_training_now
        self.model.stop_training_now = stop_training_now

    def on_epoch_end(self):
        global stop_training
        self.model.stop_training = stop_training
        if stop_training:
            self.model.info('\n  -> Train stopped by user \n\n')


# Make validation in every n training steps
class Validation(Callback):
    def __init__(self, n_steps, write_history=True, run_callbacks=True):
        self.write_history = write_history
        self.run_callbacks = run_callbacks
        self.n_steps = n_steps
        super(Callback, self).__init__()

    def on_step_end(self):
        if self.model.step % self.n_steps == 0 and self.model.step != 0:
            self.model.run_validation(write_history=self.write_history, run_callbacks=self.run_callbacks)
            do_validation = False


# Validation on keyboard press 'v'
class KeyboardValidation(Callback):
    def __init__(self, write_history=True, run_callbacks=True):
        global do_validation
        do_validation = False
        self.write_history = write_history
        self.run_callbacks = run_callbacks

        # validation
        def validation():
            global do_validation
            do_validation = True
            return True  # continue listener

        # Collect events until released
        keyboard.listen_key('v', validation)
        keyboard.start()

        print "! Note: press 'v' to run validation"
        super(Callback, self).__init__()

    def on_step_end(self):
        global do_validation
        if do_validation:
            self.model.info("\n  -> Key 'v' pressed, run validation \n")
            self.model.run_validation(write_history=self.write_history, run_callbacks=self.run_callbacks)
            do_validation = False


# Learning rate
class KeyboardLearningRate(Callback):
    def __init__(self, step_percent=0.2):

        # decrease
        def decrease_lr():
            self.model.learning_rate *= 1 - step_percent
            self.model.info('\n  -> Keyboard: Learning rate decreased to %0.2E \n\n' % self.model.learning_rate)

        # increase
        def increase_lr():
            self.model.learning_rate *= 1 + step_percent
            self.model.info('\n  -> Keyboard: Learning rate increased to %0.2E \n\n' % self.model.learning_rate)

        # Collect events until released
        keyboard.listen_key('-', decrease_lr)
        keyboard.listen_key('+', increase_lr)
        keyboard.listen_key('=', increase_lr)
        keyboard.start()

        print "! Note: press '+/=' to increase learning rate, '-' to decrease"
        super(Callback, self).__init__()


# Learning rate
class ReducingLearningRate(Callback):
    def __init__(self, monitor='val_loss', factor=0.5, patience=4,
                 verbose=0, mode='auto', epsilon=1e-10, cooldown=1, min_lr=0):

        super(Callback, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('Learning rate does not support a factor >= 1.0.')

        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # cooldown counter
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['min', 'max']:
            print 'Learning Rate mode %s is unknown, use min or max.' % self.mode
            raise ValueError('Incorrect mode in ReducingLearningRate')

        if self.mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        elif self.mode == 'max':
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_start(self):
        self._reset()

    def on_epoch_end(self):
        history = self.model.history

        # get last monitor value from history
        if self.monitor not in history:
            raise KeyError('No key in model.history: %s' % self.monitor)
        else:
            current = history[self.monitor][-1]

        # lr reducing
        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0

        elif not self.in_cooldown():
            if self.wait >= self.patience:
                old_lr = float(self.model.learning_rate)
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    self.model.learning_rate = new_lr
                    self.model.info('\n  -> Callback: Learning rate = %0.2e' % new_lr)
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
            self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0


# Testarium callback for commits during training
class Testarium(Callback):
    """
    Commits will be scored and saved after each epoch. 
    You can find your commits in testarium web & log right after the first epoch.  
    """
    def __init__(self, commit, score_func):
        self.commit = commit
        self.score_func = score_func
        super(Testarium, self).__init__()

    def on_epoch_end(self):
        try:
            desc = self.score_func(self.commit)
            self.commit.desc.update(desc)
        except Exception:
            self.model.info('\n\n! error in user score function for tfmicro.callbacks.Testarium: '
                            'your score function must be calculated during training')
            self.model.info(traceback.format_exc())

        self.commit.Save()


# Accuracy callback for logits with labels
class AccuracyCallback(Callback):
    def __init__(self):
        self.logits = None
        self.labels = None
        super(Callback, self).__init__()

    def on_start(self):
        self.model.history['accuracy'] = []
        if not hasattr(self.model, 'labels'):
            print '! warning: model must have model.labels for AccuracyCallback'
        if not hasattr(self.model, 'logits'):
            print '! warning: model must have model.logits for AccuracyCallback'

    def on_epoch_begin(self):
        self.logits = []
        self.labels = []

    def on_validation_step_end(self):
        self.logits.append(self.model.logits)
        self.labels.append(self.model.labels)

    def on_epoch_end(self):
        predicted_labels = np.argmax(np.vstack(self.logits), axis=1)
        labels = np.hstack(self.labels)
        acc = np.mean(np.equal(predicted_labels, labels))

        self.model.info('\n   > Accuracy: %0.2f\n  ' % (acc * 100))
        self.model.history['accuracy'] += [acc]

        value = summary_pb2.Summary.Value(tag="accuracy", simple_value=acc)
        global_step = self.model.epoch*self.model.data.validation_steps + self.model.valid_step
        self.model.valid_writer.add_summary(summary_pb2.Summary(value=[value]), global_step=global_step)
        self.model.valid_writer.flush()


# False Alarm & False Reject (FAFR) callback
class FafrCallback(Callback):
    def __init__(self, do_l2_norm=True, n_proc=None):
        self.embeddings = None
        self.labels = None
        self.do_l2_norm = do_l2_norm
        self.n_proc = n_proc
        self.metric = None
        self.neg = self.pos = None

        # testarium functions import on the fly
        t = __import__('testarium')
        self.fafr_parallel = t.score.fafr.fafr_parallel
        self.get_pos_neg = t.score.fafr.get_pos_neg

        # check
        if do_l2_norm and self.metric == 'hamming':
            self.model.info('\n! warning: unit lenth (l2) norm enabled but hamming distance in use')

        super(Callback, self).__init__()

    def on_start(self):
        self.model.history['thr'] = []
        self.model.history['eer'] = []
        self.model.history['minDCF'] = []

        # read config because it available
        self.metric = self.config.get('model.fafr.metric', 'cos')
        self.n_proc = self.config.get('data.n_proc', 8) if self.n_proc is None else self.n_proc

        if not hasattr(self.model, 'labels'):
            print '! warning: model must have model.labels for FafrCallback'
        if not hasattr(self.model, 'embeddings'):
            print '! warning: model must have model.embeddings for FafrCallback'

    def on_epoch_begin(self):
        self.embeddings = []
        self.labels = []

    def on_validation_step_end(self):
        self.embeddings.append(self.model.embeddings)
        self.labels.append(self.model.labels)

    def on_epoch_end(self):
        embeddings = np.vstack(self.embeddings)
        labels = np.hstack(self.labels)

        # apply l2 norm (unit len)
        if self.do_l2_norm:
            embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)

        # split scores for positives & negatives
        self.pos, self.neg = self.get_pos_neg(embeddings, embeddings, labels, labels, metric=self.metric)

        # build FAFR plots
        FA, FR, Thr = self.fafr_parallel(self.pos, self.neg, 1000, self.n_proc)
        thr = Thr[np.argmin(np.abs(FA - FR))]
        eer = FA[np.argmin(np.abs(FA - FR))]
        minDCF = np.min(100 * FA + FR)
        self.FA, self.FR, self.Thr, self.eer_thr, self.eer, self.minDCF = FA, FR, Thr, thr, eer, minDCF
        self.model.info('\n   > EER: %0.3f > minDCF: %0.3f > threshold: %0.2f\n  ' % (eer * 100, minDCF, thr), False)

        self.model.history['eer'] += [eer]
        self.model.history['minDCF'] += [minDCF]
        self.model.history['thr'] += [thr]

        value = summary_pb2.Summary.Value(tag="eer", simple_value=eer)
        global_step = self.model.epoch*self.model.data.validation_steps + self.model.valid_step
        self.model.valid_writer.add_summary(summary_pb2.Summary(value=[value]), global_step=global_step)
        self.model.valid_writer.flush()
