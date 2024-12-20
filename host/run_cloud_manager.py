import cv2
import numpy as np
import os
import os.path as osp
import torch
import argparse
import math
import copy
import time
import torch.nn.functional as F

from mmseg.apis import init_model
from mmseg.utils import register_all_modules, SampleList, dataset_aliases, get_classes, get_palette
from mmengine.config import Config, DictAction
from mmengine.runner import Runner, load_checkpoint
from mmengine.registry import MODELS, EVALUATOR, METRICS
from mmseg.models.utils import Upsample, resize
from mmseg.evaluation import IoUMetric
from mmseg.models.backbones import set_requires_grad


from io import BytesIO 
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler, ThrottledDTPHandler
from pyftpdlib.servers import FTPServer

from PIL import Image
from ftplib import FTP
import rein
import shutil

def cross_entropy(x, y):# -> torch.Tensor:
    return -(y.softmax(1) * x.log_softmax(1)).sum(1)

class BaseHandler(FTPHandler):
    def dict_to_bytes(self, state_dict):
        # Convert the image to bytes
        dict_bytes_io = BytesIO()
        torch.save(state_dict, dict_bytes_io)
        dict_bytes_io.seek(0)
        return dict_bytes_io
    
    def on_connect(self):
        print(f"Connected to {self.remote_ip}")
        self.cloud_manager.times = []

    def on_file_received(self, file_path):
        # with self.fs.open(file_path, 'rb') as file_bytes_io:
        #     image_dict = torch.load(file_bytes_io)
        # uplink_time = time.time()-image_dict['time']
        # print("Received NumPy array with shape: {}; uplink time: {}".format(image_dict['images'].shape, uplink_time))
        self.save_image(file_path)
        self.send_model()
        # self.cloud_manager.uplink_times.append(uplink_time)

    def save_image(self, file_path):
        with self.fs.open(file_path, 'rb') as file_bytes_io:
            image_dict = torch.load(file_bytes_io)
        self.num_img += 1

        img = image_dict['images'][0] # Assuming given image number is equal to 1.
        img = img.permute(1, 2, 0)
        img = Image.fromarray(img.numpy(), mode='RGB')

        tmp_path = f'tmp.png'
        img_path = os.path.join(self.cloud_manager.image_folder, f'{self.num_img:07d}.png')
        img.save(tmp_path)

        os.rename(tmp_path, img_path)
        
        num_img_save = 100
        if len(os.listdir(self.cloud_manager.image_folder)) > num_img_save:
            print(img_path.replace(f'{self.num_img:07d}', f'{self.num_img-num_img_save:07d}'))
            os.remove(img_path.replace(f'{self.num_img:07d}', f'{self.num_img-num_img_save:07d}'))
        

    def send_model(self):
        def check_model():
            path = os.path.join(self.cloud_manager.checkpoint_folder, self.cloud_manager.current_ckpt)
            if os.path.exists(path):
                if os.path.getmtime(path) == self.currmtime:
                    return False                
                self.currmtime = os.path.getmtime(path)
                print(self.currmtime)
                return True
            else:
                return False
            
        if check_model():
            print('sending model...')
            self.cloud_manager.downlink_model()
        else:
            print('model not ready.')
            self.cloud_manager.downlink_none()
        
    def on_disconnect(self):
        pass


class CloudManager():
    def __init__(self, host, port, dl_host, dl_port, image_folder_path, checkpoint_folder_path, username = 'coa', password = 'sdfsdf'):
        self.host = host
        self.port = port
        self.dl_host = dl_host
        self.dl_port = dl_port
        self.username = username
        self.password = password

        self.image_folder = image_folder_path
        if os.path.exists(self.image_folder):
            shutil.rmtree(self.image_folder)
            os.makedirs(self.image_folder)
        else:
            os.makedirs(self.image_folder)

        self.checkpoint_folder = checkpoint_folder_path
        if os.path.exists(self.checkpoint_folder):
            shutil.rmtree(self.checkpoint_folder)
            os.makedirs(self.checkpoint_folder)
        else:
            os.makedirs(self.checkpoint_folder)

        self.current_ckpt = 'ckpt.pth'

        # Set up FTP server authorizer
        authorizer = DummyAuthorizer()
        authorizer.add_user("coa", "sdfsdf", "./", perm="elradfmw")

        # Create FTP handler with the custom handler class
        dtp_handler = ThrottledDTPHandler
        dtp_handler.read_limit = 1024000 * 0 # 1Mb / sec = 1,000 Kb/sec (1000 * 1024)
        dtp_handler.write_limit = 1024000 * 0 # 1,000 Kb/sec (1000 * 1024)

        handler = BaseHandler
        handler.num_img = 0
        handler.authorizer = authorizer
        handler.dtp_handler = dtp_handler
        handler.currmtime = ''
        handler.cloud_manager = self

        self.server = FTPServer((self.host, self.port), handler)
        self.times = []
        self.uplink_times = []

        print("FTP Server running on {}:{}".format(self.host, self.port))

    def start(self):
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("FTP Server shutting down.")
            self.server.close_all()
            
    def dict_to_bytes(self, state_dict):
        # Convert the image to bytes
        dict_bytes_io = BytesIO()
        torch.save(state_dict, dict_bytes_io)
        dict_bytes_io.seek(0)
        return dict_bytes_io
    
    def get_params(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        return state_dict
    
    def downlink_model(self):
        downlink_start = time.time()
        state_dict = self.get_params(os.path.join(self.checkpoint_folder, self.current_ckpt))
        state_dict['time'] = downlink_start
        state_dict['valid'] = True
        dict_bytes_io = self.dict_to_bytes(state_dict)

        self.ftp = FTP()
        print(self.dl_host, self.dl_port)
        self.ftp.connect(self.dl_host, self.dl_port)
        self.ftp.login(self.username, self.password)
        a = self.ftp.storbinary(f"STOR {'temp'}", dict_bytes_io)
        self.ftp.quit()

    def downlink_none(self):
        downlink_start = time.time()
        state_dict = dict()
        state_dict['time'] = downlink_start
        state_dict['valid'] = False

        dict_bytes_io = self.dict_to_bytes(state_dict)

        self.ftp = FTP()
        print(self.dl_host, self.dl_port)
        self.ftp.connect(self.dl_host, self.dl_port)
        self.ftp.login(self.username, self.password)
        a = self.ftp.storbinary(f"STOR {'temp'}", dict_bytes_io)
        self.ftp.quit()

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()


    cloud_manager = CloudManager(
        host='172.27.183.243', port=9999, 
        dl_host='172.27.183.242', dl_port=9998,
        image_folder_path = './host/images',
        checkpoint_folder_path='./host/checkpoints'
        )
        
    cloud_manager.start()

if __name__ == '__main__':
    main()