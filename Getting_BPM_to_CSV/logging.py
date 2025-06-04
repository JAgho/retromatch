import serial
from datetime import datetime, timedelta
import csv

#Open a csv file and set it up to receive comma delimited input
logging = open('logging.csv',mode='a')
writer = csv.writer(logging, delimiter=",",lineterminator="\n", quoting=csv.QUOTE_MINIMAL)

#Open a serial port that is connected to an Arduino (below is Linux, Windows and Mac would be "COM4" or similar)
#No timeout specified; program will wait until all serial data is received from Arduino
#Port description will vary according to operating system. Linux will be in the form /dev/ttyXXXX
#Windows and MAC will be COMX
ser = serial.Serial(port='COM7',baudrate=115200)
ser.flushInput()

#Write out a single character encoded in utf-8; this is defalt encoding for Arduino serial comms
#This character tells the Arduino to start sending data
ser.write(bytes('x', 'utf-8'))

datetime_ser = datetime.now()
print(datetime_ser)
#runs once at first send, gets time at first run

ser_recieved = ser.readline()
decoded_bytes = ser_recieved.decode("utf-8").strip().split(',')
last_millis = int(decoded_bytes[1])


while True:
    #Read in data from Serial until \n (new line) received
    ser_bytes = ser.readline()
    #print(ser_bytes)
    
    #Convert received bytes to text format
    decoded_bytes = (ser_bytes[0:len(ser_bytes)-2].decode("utf-8"))
    print(decoded_bytes)

    decoded_bytes = decoded_bytes.split(',')
    
    ser_bytes = decoded_bytes[0] #Splits bpm from corregated string
    ser_millis = int(decoded_bytes[1]) #Time in millis from arduino start
    
    #difference in milliseconds between current and last data sent
    diff_millis = ser_millis - last_millis
    last_millis = ser_millis

    #print(ser_bytes)

    #adds the difference in ms to the current time
    datetime_ser += timedelta(milliseconds=float(diff_millis))
    
    print(datetime_ser)

    temp_time = datetime_ser.strftime('%H:%M:%S.%f')[:-3]

    
    #If Arduino has sent a string "stop", exit loop
    if (decoded_bytes == "stop"):
        break
    
    #Write received data to CSV file
    writer.writerow([datetime_ser,decoded_bytes])
            
# Close port and CSV file to exit
ser.close()
logging.close()
print("logging finished")