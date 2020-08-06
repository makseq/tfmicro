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
from __future__ import print_function

import os
import time
import multiprocessing
import queue as Queue

class Workers:
    def __init__(self, c, mode, data):
        # check openblas threads to prevent threads hell while using multiprocessing
        if 'OPENBLAS_NUM_THREADS' not in os.environ or int(os.environ['OPENBLAS_NUM_THREADS']) != 1:
            print('! tfmicro.workers error: Set OPENBLAS_NUM_THREADS=1 before import numpy or numpy depending modules')
            exit(-1)

        self.c = c
        self.data = data
        self.mode = mode
        self.verbose = c['data.verbose']
        self.maxsize = c['data.queue_size']

        self.state = ''
        self.manager = multiprocessing.Manager()
        self.queue_in = self.manager.Queue()
        self.queue_out = self.manager.Queue(maxsize=self.maxsize)
        self.jobs = []

    def debug(self, *args):
        if self.verbose > 0:
            for a in args:
                print(a, end=' ')
            print()

    @staticmethod
    def clear_queue(queue):
        try:
            while True:
                queue.get_nowait()
        except Queue.Empty:
            pass

    def start(self):
        # check if here is no progress with this instance of workers
        if self.state:
            raise Exception("Call generator_stop('" + self.mode + "') before start new one")
        # set progress
        self.state = 'progress'

        n_processes = self.c['data.n_proc']
        jobs = [multiprocessing.Process(target=self.task, args=(self.queue_in, self.queue_out, id_))
                for id_ in range(n_processes)]
        for j in jobs:
            j.daemon = True
        [j.start() for j in jobs]
        self.jobs = jobs

    def task(self, q_in, q_out, proc_id):
        # check if 'mode' & 'proc_id' arguments in evaluate_batch_prepare implementation
        kwargs = {}
        if 'mode' in self.data.evaluate_batch_prepare.__code__.co_varnames:
            kwargs.update({'mode': self.mode})
        if 'proc_id' in self.data.evaluate_batch_prepare.__code__.co_varnames:
            kwargs.update({'proc_id': proc_id})
        prepared = self.data.evaluate_batch_prepare(**kwargs)

        while True:
            # read
            s = time.time()
            w = q_in.get()
            self.debug(' >', self.mode, ': pid', os.getpid(), 'get', time.time() - s)

            # exit
            if w is None:
                q_out.put(None)
                self.debug(' -->', self.mode, ': pid', os.getpid(), 'exited')
                return

            # evaluate
            s = time.time()
            result = self.data.evaluate_batch(w, prepared)
            self.debug(' >', self.mode, ': pid', os.getpid(), 'compute', time.time() - s)

            # write
            s = time.time()
            q_out.put(result)
            self.debug(' <', self.mode, ': pid', os.getpid(), 'put', time.time() - s)

    def stop(self):
        self.debug('! ---> try to exit from jobs', self.mode)
        self.state = 'stop'  # set stop signal

        # clear queue_in
        self.clear_queue(self.queue_in)
        [self.queue_in.put(None) for _ in self.jobs]

        # clear queue_out by getting None
        none = 0
        while none < len(self.jobs):
            result = self.queue_out.get()
            none += result is None
            self.debug(self.mode, ': none count', none,
                       'q_in size', self.queue_out.qsize(), 'q_out size', self.queue_in.qsize())

        # join & state reset
        self.debug(self.jobs)
        [j.join() for j in self.jobs]  # wait until jobs exits
        self.state = ''  # reset
        self.debug('! ---> successfully exited from jobs', self.mode)

    def get(self):
        result = self.queue_out.get()
        self.debug(self.mode, ' <<< queue_out size', self.queue_out.qsize())
        return result

    def put(self, w):
        self.queue_in.put(w)
        self.debug(self.mode, '\n >>> new queue_in size', self.queue_in.qsize(), '\n')

    def get_loading(self):
        return self.queue_out.qsize()
