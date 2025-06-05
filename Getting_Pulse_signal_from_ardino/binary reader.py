import struct


with open("binary_stream.bin", "rb") as binary_file:
    while True:
        chunk = binary_file.read(12)  # Read 12 bytes (3 unsigned longs)
        if len(chunk) < 12:
            break  # EOF reached or incomplete chunk

        signal, timestamp, funnyno = struct.unpack('<LLL', chunk)
        print(f"Signal: {signal}, Funny Number: {funnyno}, Time: {timestamp}")




