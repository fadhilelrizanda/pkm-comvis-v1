import object_tracker_1
import object_tracker_2
import threading


if __name__ == '__main__':
    thread1 = threading.Thread(target=object_tracker_1.main())
    thread2 = threading.Thread(target=object_tracker_2.main())
    thread1.start()
    thread2.start()
