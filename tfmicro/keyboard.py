import sys
import tty
import time
import termios
import threading

global started, keyboard_events
keyboard_events = {}
started = False


def reset():
    keyboard_events = {}


def listen_key(key, action):
    keyboard_events.update({key: action})


def start():
    global started
    if started:
        return
    started = True

    def read_inp():
        global keyboard_events
        while True:
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                answer = sys.stdin.read(1)
                k = keyboard_events
                if answer in k:
                    continue_ = k[answer]()
                    if continue_ is not None and continue_ == False:
                        keyboard_events = {key: value for key, value in keyboard_events.items() if key != answer}
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

            time.sleep(0.01)

    t = threading.Thread(target=read_inp)

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

