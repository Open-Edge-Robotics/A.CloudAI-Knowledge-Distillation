from ftplib import FTP
import cv2
import numpy as np
from io import BytesIO
import torch
import time

from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler, ThrottledDTPHandler
from pyftpdlib.servers import FTPServer

import threading


class MyHandler(FTPHandler):
    def on_connect(self):
        pass

    def on_file_received(self, file_path):
        with self.fs.open(file_path, 'rb') as file_bytes_io:
            self.device_manager.dl_state_dict = torch.load(file_bytes_io)
        self.device_manager.dl_is_finished = True
        self.device_manager.is_in_progress = False

    def on_disconnect(self):
        pass

class DeviceManager():
    def __init__(self, ul_host, ul_port, dl_host, dl_port, bpsspeed=0, username = 'coa', password = 'sdfsdf'):
        self.host = ul_host
        self.port = ul_port
        self.myip = dl_host
        self.myport = dl_port

        self.username = username
        self.password = password
        self.image_send_limit = 1

        self.dl_is_finished = False
        self.is_in_progress = False

        # This is for uplink (device->cloud).
        self.ftp = FTP()
        self.ftp.connect(self.host, self.port)
        self.ftp.login(self.username, self.password)
        print('Client is connected with Server [ SERVER_IP: ', self.port, ', SERVER_PORT: ', str(self.port), ']')

        # This is for downlink (cloud->device).
        # Set up FTP server authorizer
        authorizer = DummyAuthorizer()
        authorizer.add_user("coa", "sdfsdf", "./", perm="elradfmw")

        # Create FTP handler with the custom handler class
        dtp_handler = ThrottledDTPHandler
        dtp_handler.read_limit = 1024000 * bpsspeed #1024000  # 1Mb / sec = 1,000 Kb/sec (1000 * 1024)
        dtp_handler.write_limit = 1024000 * bpsspeed #1024000  # 1,000 Kb/sec (1000 * 1024)

        handler = MyHandler
        handler.authorizer = authorizer
        handler.dtp_handler = dtp_handler
        handler.device_manager = self

        self.server_downlink = FTPServer((self.myip, self.myport), handler)
        try:
            th = threading.Thread(target=self._start)
            th.daemon = True
            th.start()
        except KeyboardInterrupt:
            print("Failed to generate FTP server for downlink. Please check network connection.")

    def _start(self):
        try:
            self.server_downlink.serve_forever()
        except KeyboardInterrupt:
            print("FTP Server shutting down.")
            self.server.close_all()

    def uplink(self, image_array):
        self.is_in_progress = True
        try:
            self.th = threading.Thread(target=self._send_image_array, args=([image_array,]))
            self.th.daemon = True
            self.th.start()
        except KeyboardInterrupt:
            print("Failed to send image array. Please check network connection.")

    def update(self, model):
        # Update the model with downlinked model gradients.
        # model = model + downlinked gradients
        is_updated = self.model_update_base(model)
        self.dl_is_finished = False
        self.downlink_time = time.time() - self.dl_state_dict['time']
        return is_updated, model
    
    def model_update_base(self, model):
        # self.device_manager.dl_state_dict
        # print(self.dl_state_dict['valid'])
        if self.dl_state_dict['valid']:
            delta_params = self.dl_state_dict['params']
            model.load_state_dict(delta_params, strict=True)
            return 1
        else:
            return 0

    def _send_image_array(self, image_array):
        image_dict = dict()
        image_dict['images']=image_array
        image_dict['time']  = time.time()
        img_bytes_io = self.image_to_bytes(image_dict)
        self.ftp.storbinary(f"STOR {'temp'}", img_bytes_io)

    def image_to_bytes(self, image_dict):
        # Convert the image to bytes
        img_bytes_io = BytesIO()
        # np.save(img_bytes_io, image)
        torch.save(image_dict, img_bytes_io)
        img_bytes_io.seek(0)
        return img_bytes_io

    def join(self):
        self.th.join()
    

# class FTPClient:
#     def __init__(self, host, port, username = 'coa', password = 'sdfsdf'):
#         self.host = host
#         self.port = port
#         self.username = username
#         self.password = password
#         self.is_state_dict = False

#         HOST = '210.125.85.243' # Replace with your server's IP or hostname
#         PORT = 9998 # 
#         # Set up FTP server authorizer
#         authorizer = DummyAuthorizer()
#         authorizer.add_user("coa", "sdfsdf", "./", perm="elradfmw")

#         # Create FTP handler with the custom handler class
#         dtp_handler = ThrottledDTPHandler
#         dtp_handler.read_limit = 1024000 * 50  # 1Mb / sec = 1,000 Kb/sec (1000 * 1024)
#         dtp_handler.write_limit = 1024000 * 50  # 1,000 Kb/sec (1000 * 1024)

#         self.handler = MyHandler
#         self.handler.authorizer = authorizer
#         self.handler.dtp_handler = dtp_handler
#         self.handler.client = self

#         server = FTPServer((HOST, PORT), self.handler)
        
#         print("FTP Server running on {}:{}".format(HOST, PORT))

#         try:
#             th = threading.Thread(target=server.serve_forever)
#             th.start()
#         except KeyboardInterrupt:
#             print("FTP Server shutting down.")
#             server.close_all()

#         self.ftp = FTP()
#         self.ftp.connect(self.host, self.port)
#         self.ftp.login(self.username, self.password)
#         print(u'Client is connected with Server [ SERVER_IP: ', self.port, ', SERVER_PORT: ', str(self.port), ']')

#     def send_image(self, image):
#         img_bytes_io = self.image_to_bytes(image)

#         self.ftp.storbinary(f"STOR {'temp'}", img_bytes_io)


#     def image_to_bytes(self, image):
#         # Convert the image to bytes
#         img_bytes_io = BytesIO()
#         np.save(img_bytes_io, image)
#         img_bytes_io.seek(0)
#         return img_bytes_io