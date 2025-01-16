import socket
import cv2
import pickle

s = socket.socket()
host = "rasp3"
port = 2345
s.connect((host, port))


def readF():
    while True:
        data = b""
        while True:
            packet = s.recv(4096)
            data += packet
            if len(data) >= 921764:
                break

        s.send("ack".encode("utf-8"))
        data_arr = pickle.loads(data)
        cv2.imshow("frame", data_arr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            #break
            pass


readF()
