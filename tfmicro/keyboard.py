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

global started, keyboard_events, exiter
keyboard_events = {}
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
        self.old = termios.tcgetattr(self.fd)

    def backward(self):
        if no_keyboard: return
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)


exiter = Exiter()


def reset():
    keyboard_events = {}


def listen_key(key, action):
    keyboard_events.update({key: action})


def start():
    global started, stop

    if started or no_keyboard:
        return

    def run():
        global keyboard_events, exiter

        while not exiter.stop:
            try:
                exiter.forward()
                tty.setcbreak(exiter.fd)
                answer = sys.stdin.read(1)
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

