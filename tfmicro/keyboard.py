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

import sys
import time
try:
    import termios
    import tty
    no_keyboard = False
except:
    no_keyboard = True
    print '!!! Warning: keyboard callbacks is not supported!'

import threading

global started, keyboard_events, exiter, pressed
keyboard_events = {}
pressed = []
started = False


class Exiter():
    def __init__(self):
        self.stop = False
        self.forward()
        self.backward()

    def __del__(self):
        self.stop = True
        self.backward()

    def forward(self):
        if no_keyboard: return
        self.fd = sys.stdin.fileno()
        try:
            self.old = termios.tcgetattr(self.fd)
        except Exception as e:
            pass

    def backward(self):
        if no_keyboard: return
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)
        except Exception as e:
            pass


exiter = Exiter()


def reset():
    keyboard_events = {}
    pressed = []


def listen_key(key, action):
    keyboard_events.update({key: action})


# use this to check some key was pressed
def was_pressed(key):
    global pressed
    if key in pressed:
        pressed.remove(key)
        return True
    return False


def start():
    global started, stop

    if started or no_keyboard:
        return

    def run():
        global keyboard_events, exiter, pressed

        while not exiter.stop:
            try:
                exiter.forward()
                tty.setcbreak(exiter.fd)
                answer = sys.stdin.read(1)
                pressed += [answer]
                k = keyboard_events
                if answer in k:
                    continue_ = k[answer]()
                    if continue_ is not None and continue_ == False:
                        keyboard_events = {key: value for key, value in keyboard_events.items() if key != answer}
            finally:
                exiter.backward()

            time.sleep(0.01)

    started = True
    t = threading.Thread(target=run)
    t.setDaemon(True)
    t.start()


if __name__ == '__main__':

    def my_func():
        print 'Quit!'

    def my_func2():
        print 'Try!'
        return False

    start()
    listen_key('q', my_func)
    listen_key('t', my_func2)

    while True:
        print 'step'
        time.sleep(5)

