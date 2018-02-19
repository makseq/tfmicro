import sys

import tensorflow as tf

import keyboard

stop_training = False


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
            global stop_training
            stop_training = True
            sys.stdout.write('\n  -> Train will stop at the end of epoch! \n')
            return False  # stop listener

        # Collect events until released
        keyboard.listen_key('q', on_stop)
        keyboard.start()

        print "! Note: press 'q' to stop training"
        super(Callback, self).__init__()

    def on_epoch_end(self):
        global stop_training
        self.model.stop_training = stop_training
        if stop_training:
            sys.stdout.write(' -> Train stopped by user ')


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


'''# Callback: Early stopping by crossing loss vs val_loss (cross validation)
class EarlyStoppingByCrossing(Callback):
    def __init__(self, epsilon):
        super(Callback, self).__init__()
        self.eps = epsilon

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        valid = logs.get('val_loss')
        if loss < valid + self.eps:
            print("Epoch %05d: early stopping loss < cross_validation" % epoch)
            self.model.stop_training = True


# Callback: Reset states
class ResetStates(Callback):
    def __init__(self):
        super(Callback, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        if c['model.stateful']:
            self.model.reset_states()
reset_states = ResetStates()'''

