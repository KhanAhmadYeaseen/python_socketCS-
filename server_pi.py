import socket
import time
import threading
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import pickle
import struct
import json

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

time.sleep(0.1)

host='0.0.0.0'
port=2345
s=socket.socket()
s.bind((host,port))
s.listen(2)
conn,addr=s.accept()
print("Connected by",addr)

def connecting():
    global conn
    conn.close()
    conn,addr=s.accept()
    print("Connected by",addr)
    

def writeF():
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        rawCapture.truncate(0)
        try:
            data = pickle.dumps(image)
            conn.send(data)
            while True:
                k = conn.recv(1024)
                if k.decode("utf-8") == "ack":
                    break
        except ConnectionResetError as e:
            print("Connection stopped, waiting for new connection")
            connecting()

writeF()

