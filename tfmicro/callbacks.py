import sys
import numpy as np
import tensorflow as tf
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

    def on_validation_step_begin(self): pass

    def on_validation_step_end(self): pass

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
        self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

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
                sys.stdout.write('\n  -> Key \'q\' pressed twice. Train will stop at the end of step! \n')
                return False  # stop listener
            # first press
            else:
                stop_training = True
                sys.stdout.write('\n  -> Train will stop at the end of epoch! \n')
                return True # stop listener

        # Collect events until released
        keyboard.listen_key('q', on_stop)
        keyboard.start()

        print "! Note: press 'q' to stop training"
        print "! Note: press 'q' twice to stop training after the step"
        super(Callback, self).__init__()

    def on_step_end(self):
        global stop_training_now
        self.model.stop_training_now = stop_training_now

    def on_epoch_end(self):
        global stop_training
        self.model.stop_training = stop_training
        if stop_training:
            sys.stdout.write('\n -> Train stopped by user \n')


# Learning rate
class KeyboardLearningRate(Callback):
    def __init__(self, step_percent=0.2):

        # decrease
        def decrease_lr():
            self.model.learning_rate *= 1 - step_percent
            sys.stdout.write('\n  -> Keyboard: Learning rate decreased to %0.2E \n' % self.model.learning_rate)

        # increase
        def increase_lr():
            self.model.learning_rate *= 1 + step_percent
            sys.stdout.write('\n  -> Keyboard: Learning rate increased to %0.2E \n' % self.model.learning_rate)

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
                    print('\n  -> Callback: Learning rate = %0.2e' % new_lr)
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
            self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0


