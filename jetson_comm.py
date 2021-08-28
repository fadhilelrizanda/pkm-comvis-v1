# import serial
# import time

# esp32 = serial.Serial(
# port = "/dev/ttyUSB0",
# baudrate = 115200,
# bytesize = serial.EIGHTBITS, 
# parity = serial.PARITY_NONE,
# stopbits = serial.STOPBITS_ONE, 
# timeout = 1,
# xonxoff = False,
# rtscts = False,
# dsrdtr = False,
# writeTimeout = 2
# )
# esp32.rtscts = True
# esp32.dsrdtr = True

# Importing Libraries
import serial
import time
arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.1)
def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data
while True:
    num = input("Enter a number: ") # Taking input from user
    value = write_read(num)
    print(value) # printing the value