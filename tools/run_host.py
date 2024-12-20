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

from ftplib import FTP
import rein

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
        with self.fs.open(file_path, 'rb') as file_bytes_io:
            image_dict = torch.load(file_bytes_io)
        uplink_time = time.time()-image_dict['time']

        print("Received NumPy array with shape: {}; uplink time: {}".format(image_dict['images'].shape, uplink_time))
        self.cloud_manager.uplink_times.append(uplink_time)

        self.cloud_manager.run_distillation_pl(image_dict['images'])#image_dict['images'])

    def on_disconnect(self):
        pass


class CloudManager():
    def __init__(self, host, port, dl_host, dl_port, cloud_model, device_model, source_buffer, username = 'coa', password = 'sdfsdf'):
        self.host = host
        self.port = port
        self.dl_host = dl_host
        self.dl_port = dl_port
        self.username = username
        self.password = password

        self.cloud_model = cloud_model
        self.device_model = device_model
        self.source_buffer = source_buffer

        self.resize_shape_device = (512, 1024)
        self.resize_shape_cloud = (1024, 1024)

        params = []
        names = []
        for name, param in self.device_model.named_parameters():
            if param.requires_grad: #
                params.append(param)
                names.append(name)
        print(names)
        self.optimizer = torch.optim.Adam(params, lr=0.00008 / 8, betas=(0.9, 0.999))

        # Set up FTP server authorizer
        authorizer = DummyAuthorizer()
        authorizer.add_user("coa", "sdfsdf", "./", perm="elradfmw")

        # Create FTP handler with the custom handler class
        dtp_handler = ThrottledDTPHandler
        dtp_handler.read_limit = 1024000 * 0 # 1Mb / sec = 1,000 Kb/sec (1000 * 1024)
        dtp_handler.write_limit = 1024000 * 0  # 1,000 Kb/sec (1000 * 1024)

        handler = BaseHandler
        handler.authorizer = authorizer
        handler.dtp_handler = dtp_handler
        handler.cloud_manager = self

        self.server = FTPServer((self.host, self.port), handler)
        self.times = []
        self.uplink_times = []

        print("FTP Server running on {}:{}".format(self.host, self.port))

    def run_distillation_pl(self, image_array):
        # Distillation abstraction function need to be implemented.
        dis_start = time.time()
        self.device_model_previous = copy.deepcopy(self.device_model)

        self.optimizer.zero_grad()
        x = image_array.cuda()

        foutputs_ = []
        with torch.no_grad():
            img = x #resize(x.float(), (2048, 1024), mode = 'bilinear').int()#self.resize_shape_cloud)
            # print(img)
            data = dict()
            data['inputs'] = img
            outputs =  self.cloud_model.test_step(data)
            foutputs_ = torch.stack([outputs[i].seg_logits.data.max(0).indices for i in range(len(x))])

        # img = resize(x, self.resize_shape_device, mode = 'bilinear') #self.resize_shape_cloud)
        data = dict()
        data['inputs'] = x
        outputs = self.device_model.test_step(data)
        outputs = torch.stack([outputs[i].seg_logits.data for i in range(len(x))])
        print(foutputs_.shape, outputs.shape)

        # outputs = resize(outputs, size = foutputs_.size()[1:])
        loss = F.cross_entropy(outputs, foutputs_).mean()
        loss.backward()
        self.optimizer.step()        
        dis_end = time.time()
        dis_time = dis_end - dis_start
        self.times.append(dis_time)
        print("Edge model is adjusted by the cloud model; time-cost: {}".format(dis_time))
        self.downlink()
        print("Send delta params back to the client.")

    def dict_to_bytes(self, state_dict):
        # Convert the image to bytes
        dict_bytes_io = BytesIO()
        torch.save(state_dict, dict_bytes_io)
        dict_bytes_io.seek(0)
        return dict_bytes_io
    
    def downlink(self):
        downlink_start = time.time()
        state_dict = self.get_params()
        state_dict['time'] = downlink_start
        dict_bytes_io = self.dict_to_bytes(state_dict)

        self.ftp = FTP()
        print(self.dl_host, self.dl_port)
        self.ftp.connect(self.dl_host, self.dl_port)
        self.ftp.login(self.username, self.password)
        a = self.ftp.storbinary(f"STOR {'temp'}", dict_bytes_io)
        self.ftp.quit()

    def start(self):
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            print("FTP Server shutting down.")
            self.server.close_all()

    def get_params(self):
        state_dict = self.device_model.state_dict()
        delta_model = {}

        # for name, param in self.device_model.named_parameters():
        for k, key in enumerate(state_dict):
            if 'adaptformer' in key:
                values = (state_dict[key])
                delta_model[key] = values
        state_dict ={'params': delta_model}
        return state_dict

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('config_cm', help='train config file path')
    parser.add_argument('checkpoint_cm', help='checkpoint file')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    print('CONFIG: ', args.config, args.config_cm)
    print('CHECKPOINT: ', args.checkpoint, args.checkpoint_cm)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint

    cfg_cm = Config.fromfile(args.config_cm)
    cfg_cm.load_from = args.checkpoint_cm

    register_all_modules()
    device = 'cuda:0'

    cloud_model = MODELS.build(cfg_cm.model)
    checkpoint = load_checkpoint(cloud_model.backbone, './checkpoints/dinov2_converted_1024x1024.pth', map_location='cpu')
    checkpoint = load_checkpoint(cloud_model, cfg_cm.load_from, map_location='cpu')
    cloud_model.dataset_meta = {
                'classes': get_classes('cityscapes'),
                'palette': get_palette('cityscapes')
            }
    cloud_model.to(device)
    cloud_model.eval()

    device_model = MODELS.build(cfg.model)
    checkpoint = load_checkpoint(device_model, cfg.load_from, map_location='cpu')
    device_model.dataset_meta = {
                'classes': get_classes('cityscapes'),
                'palette': get_palette('cityscapes')
            }
    set_requires_grad(device_model, ['adaptformer'])
    

    testloader_clear = Runner.build_dataloader(cfg['train_dataloader'])

    device_model.to(device)
    device_model.eval()


    cloud_manager = CloudManager(
        host='192.168.0.10', port=9999, 
        dl_host='192.168.0.20', dl_port=9998,
        cloud_model = cloud_model, device_model = device_model,
        source_buffer = testloader_clear.dataset)
    cloud_manager.start()

if __name__ == '__main__':
    main()