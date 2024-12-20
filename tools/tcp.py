import sys
import socket
import base64
import numpy as np
import time
import datetime

from _thread import *

class ClientSocket:
    def __init__(self, ip, port=9999):
        self.TCP_SERVER_IP = ip
        self.TCP_SERVER_PORT = port
        self.connectCount = 0
        self.connectServer()

    def connectServer(self):
        try:
            self.sock = socket.socket()
            self.sock.connect((self.TCP_SERVER_IP, self.TCP_SERVER_PORT))
            print(u'Client socket is connected with Server socket [ TCP_SERVER_IP: ' + self.TCP_SERVER_IP + ', TCP_SERVER_PORT: ' + str(self.TCP_SERVER_PORT) + ' ]')
            self.connectCount = 0
            # self.sendImage()
        except Exception as e:
            print(e)
            self.connectCount += 1
            time.sleep(1)
            if self.connectCount == 10:
                print(u'Connect fail %d times. exit program'%(self.connectCount))
                sys.exit()
            print(u'%d times try to connect with server'%(self.connectCount))
            self.connectServer()

    def sendImage(self, image, cnt):
        image = image

        try:
            # stime = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
                
            data = np.array(image)
            stringData = data.tostring() #base64.b64encode(image)
            # print(data.dtype, data.shape)
            length = str(len(stringData))
            shape = '{},{},{},'.format(data.shape[1], data.shape[2], data.shape[3])
            # print(length)

            self.sock.sendall(length.encode('utf-8').ljust(64))
            self.sock.sendall(shape.encode('utf-8').ljust(64))
            self.sock.sendall(stringData)
            # self.sock.send(stime.encode('utf-8').ljust(64))
            print('send time of ', cnt, ' image: ', datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'), data.shape)
        except Exception as e:
            print(e)
            self.sock.close()
            time.sleep(1)
            self.connectServer()

def start_client(host_ip, port=9999):
    HOST = host_ip#'210.125.85.246' ## ip address for server ##
    PORT = port#9999


    client_socket = ClientSocket(HOST, PORT)
    return client_socket
