'''
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
'''

import matplotlib.pyplot as plt
import numpy as np

import callbacks


def checker(func):
    def _decorator(self, *args, **kwargs):
        if self.config['callbacks.plot']:
            func(self, *args, **kwargs)
    return _decorator


class PlotCallback(callbacks.Callback):
    @checker
    def on_start(self):
        plt.ion()
        self.n = self.config['callbacks.plot.n']
        self.f, self.ax = plt.subplots(self.n, 1, figsize=(5, 8))

    @checker
    def on_finish(self):
        plt.ioff()
        plt.close()

    @checker
    def on_epoch_begin(self):
        self.step = 0
        self.y = []
        self.preds = []
        self.train_preds = []
        self.train_y = []
        self.train_step = 0

    @checker
    def on_validation_step_end(self):
        self.preds += [self.model.test_prediction]
        self.y += [self.model.test_y]

    @checker
    def on_step_end(self):
        if len(self.train_preds) < 3 and self.train_step % 10 == 0:
            self.train_preds += [self.model.train_prediction[0:1]]  # take only the first
            self.train_y += [self.model.y[0:1]]  # take only the first
        self.train_step += 1

    @checker
    def on_epoch_end(self):
        n, ax = self.n, self.ax
        # validation
        self.y = np.vstack(self.y)
        self.preds = np.vstack(self.preds)
        # train
        self.train_y = np.vstack(self.train_y)
        self.train_preds = np.vstack(self.train_preds)
        # train + validation
        self.y = np.vstack([self.train_y, self.y])
        self.preds = np.vstack([self.train_preds, self.preds])

        if self.step != 0:
            return

        k = self.y.shape[0]
        for i in range(n if n < k else k):
            ax[i].clear()

            y = self.y[i]  # np.concatenate([self.model.test_x[i], self.model.test_y[i]])
            pred = self.preds[i]

            y = self.model.data.unroll(y, None)
            pred = self.model.data.unroll(pred, None)

            min = np.min(pred[:, 0])
            max = np.max(pred[:, 1])
            argmin = np.argmin(pred[:, 0])
            argmax = np.argmax(pred[:, 1])

            ax[i].plot(pred[:, 0], color='b')
            ax[i].plot(pred[:, 1], color='g')
            ax[i].scatter([0], [min], color='b', alpha=1, marker='>')
            ax[i].scatter([0], [max], color='g', alpha=1, marker='>')

            min = np.min(y[:, 0])
            max = np.max(y[:, 1])
            argmin = np.argmin(y[:, 0])
            argmax = np.argmax(y[:, 1])

            ax[i].plot(y[:, 0], color='y', alpha=0.5)
            ax[i].plot(y[:, 1], color='r', alpha=0.5)
            ax[i].scatter([0], [min], color='y', alpha=1, marker='+')
            ax[i].scatter([0], [max], color='r', alpha=1, marker='+')

        self.f.tight_layout()
        self.f.canvas.draw()
        plt.pause(0.0001)
        self.step += 1


class PlotCallbackStates(callbacks.Callback):
    def on_start(self):
        plt.ion()
        self.n = 7
        self.f, self.ax = plt.subplots(1, 1, figsize=(5, 2))
        self.step = 0

    def on_finish(self):
        plt.ioff()
        plt.close()

    def on_epoch_begin(self):
        self.step = 0

    def on_step_end(self):
        if self.step % 10 == 0:
            ax = self.ax
            img = np.vstack([self.model.lstm_states.c[0].flatten(), self.model.lstm_states.h[0].flatten()])
            ax.imshow(img, interpolation='nearest', aspect='auto', cmap='gray')
            self.f.canvas.draw()
            plt.pause(0.0001)

        self.step += 1