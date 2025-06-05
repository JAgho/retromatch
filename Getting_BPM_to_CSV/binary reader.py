import struct


with open("binary_stream.bin", "rb") as binary_file:
    lines = binary_file.readlines()
    for i in lines:
        print(i)
        signal, timestamp, funnyno = struct.unpack('<LLL', i)




