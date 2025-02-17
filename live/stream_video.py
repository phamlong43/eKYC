import cv2
import socket
import struct
from gpiozero import LED

led = LED(18)
led2 = LED(17)

# Configuration
server_ip = '172.16.248.220'   # Laptop or receiver
server_port = 8554          # Receiver port

# Open Pi Camera
cap = cv2.VideoCapture(0)  # Use camera device (default is 0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set up socket connection
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Encode frame
    _, encoded_frame = cv2.imencode('.jpg', frame)

    # Send the size of the frame followed by the frame
    frame_size = len(encoded_frame)
    client_socket.sendall(struct.pack("L", frame_size) + encoded_frame.tobytes())
    response = client_socket.recv(1)  # Expecting 1 byte of response
    if response:
        response_value = int.from_bytes(response, byteorder='big')
        if response_value == 1:
            led.on()
            led2.off()
        if response_value == 0:
            led.off()
            led2.off()
        if response_value == 2:
            led.off()
            led2.on()


cap.release()
client_socket.close()
