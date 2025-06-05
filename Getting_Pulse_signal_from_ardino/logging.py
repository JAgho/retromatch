import serial
from datetime import datetime, timedelta
import csv
from bitarray import bitarray
import struct

#Open a csv file and set it up to receive comma delimited input
logging = open('logging.csv',mode='a')
writer = csv.writer(logging, delimiter=",",lineterminator="\n", quoting=csv.QUOTE_MINIMAL)

#Open a serial port that is connected to an Arduino (below is Linux, Windows and Mac would be "COM4" or similar)
#No timeout specified; program will wait until all serial data is received from Arduino
#Port description will vary according to operating system. Linux will be in the form /dev/ttyXXXX
#Windows and MAC will be COMX
ser = serial.Serial(port='COM3',baudrate=115200)

#Write out a single character encoded in utf-8; this is defalt encoding for Arduino serial comms
#This character tells the Arduino to start sending data
ser.write(bytes('x', 'utf-8'))

datetime_ser = datetime.now()
ser.flushInput()
ser.flushOutput()

file =  open("binary_stream.bin", "wb")
logs = 0

try :
        while (True):
            if ser.in_waiting >= 12:
                data = ser.read_until(expected= b'~\x10\x00\x00', size= 12)
                #print("Raw bytes:", " ".join(f"{b:02X}" for b in data))
                file.write(data)
                # Unpack 3 unsigned longs (little-endian): Signal, FunnyNo, Time
                signal, timestamp, funnyno = struct.unpack('<LLL', data)
                logs += 1
                if logs % 50 == 0:
                    print(f"Signal: {signal}, Time: {timestamp},Funny Number: {funnyno}")

except KeyboardInterrupt:
    print("Closing")
    ser.close()

finally:
    file.close()
    print("logging finished")