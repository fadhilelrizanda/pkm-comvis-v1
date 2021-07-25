import threading
import time


def infiniteloop1():
    while True:
        print('Loop 1')
        time.sleep(1)


def infiniteloop2():
    while True:
        print('Loop 2')
        time.sleep(1)


thread1 = threading.Thread(target=infiniteloop1)
thread1.start()

thread2 = threading.Thread(target=infiniteloop2)
thread2.start()
