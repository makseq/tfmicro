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
import threading
import time


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
        self.i = -1

    def __iter__(self):
        return self

    def __next__(self):  # Py3
        with self.lock:
            self.i += 1
            return self.i, next(self.it)

    def next(self):  # Py2
        with self.lock:
            self.i += 1
            return self.i, self.it.next()


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


class threadsafe_iter_advanced:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()
        self.i = -1

    def __iter__(self):
        return self

    def __next__(self):  # Py3
        with self.lock:
            self.i += 1
            return self.i, next(self.it)

    def next(self):  # Py2
        with self.lock:
            self.i += 1
            return self.i, self.it.next()


def threadsafe_generator_advanced(f):
    def g(*a, **kw):
        return threadsafe_iter_advanced(f(*a, **kw))

    return g


class ThreadedGenerator(object):
    def __init__(self, data, mode, max_queue_size=1000, thread_num=4, debug=0):
        self.q = {}
        self.last_out = 0
        self.lock = threading.Lock()
        self.max_queue_size = max_queue_size
        self.stop_threads = False
        self.verbose = debug

        self.data = data
        self.mode = mode
        self.generator = data.generator(mode)
        self.threads = None
        self.thread_num = thread_num

    def start(self):
        self.stop_threads = False

        if self.thread_num > 0:
            self.threads = [threading.Thread(target=self.task) for _ in range(self.thread_num)]
            [t.setDaemon(True) for t in self.threads]  # correct exit if main thread is closed
            [t.start() for t in self.threads]

        return self

    def debug(self, *args):
        if self.verbose > 0:
            for a in args:
                print(a,)
            print()

    def stop(self):
        self.stop_threads = True
        # stop generator
        self.data.generator_stop(self.mode)
        # stop main thread
        if self.thread_num > 0:
            [t.join() for t in self.threads]

    def get_values(self):
        # threaded version
        if self.thread_num > 0:
            self.debug('get', self.last_out)

            # wait for last_out
            while self.last_out not in self.q:
                time.sleep(0.001)

            result = self.q.pop(self.last_out)  # pop
            self.last_out += 1
            return result

        # no threads
        else:
            for i, value in self.generator:
                return value

    def task(self):
        for i, values in self.generator:

            while len(self.q) >= self.max_queue_size and self.last_out < i:
                time.sleep(0.001)
                if self.stop_threads:
                    return

            if self.thread_num > 0:
                self.lock.acquire(True)

            self.q[i] = values  # push
            self.debug('put', len(self.q), i)

            if self.thread_num > 0:
                self.lock.release()

            if self.stop_threads:
                return
