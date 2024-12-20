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
from copy import deepcopy

def cross_entropy(x, y):# -> torch.Tensor:
    return -(y.softmax(1) * x.log_softmax(1)).sum(1)

def create_ema_model(model):
    ema_model = deepcopy(model)#get_model(args.model)(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher=0.999, iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    if True:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param [:].data[:]

class BaseHandler(FTPHandler):
    def dict_to_bytes(self, state_dict):
        # Convert the image to bytes
        dict_bytes_io = BytesIO()
        torch.save(state_dict, dict_bytes_io)
        dict_bytes_io.seek(0)
        return dict_bytes_io
    
    def send_delta_params(self):
        # Send data to the client
        state_dict = self.edge_model.state_dict()
        dict_bytes_io = self.dict_to_bytes(state_dict)

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
        self.device_model_ema = create_ema_model(device_model)
        self.source_buffer = source_buffer

        self.resize_shape_device = (512, 1024)
        self.resize_shape_cloud = (1024, 1024)

        params = []
        names = []
        for name, param in self.device_model.named_parameters():
            if param.requires_grad and 'adapt' in name: #
                params.append(param)
                names.append(name)
        print(names)
        self.optimizer = torch.optim.Adam(params, lr=0.00006 / 8, betas=(0.9, 0.999), weight_decay = 1e-4)

        # Set up FTP server authorizer
        authorizer = DummyAuthorizer()
        authorizer.add_user("coa", "sdfsdf", "./", perm="elradfmw")

        # Create FTP handler with the custom handler class
        dtp_handler = ThrottledDTPHandler
        dtp_handler.read_limit = 1024000 * 11 # 1Mb / sec = 1,000 Kb/sec (1000 * 1024)
        dtp_handler.write_limit = 1024000 * 11  # 1,000 Kb/sec (1000 * 1024)
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
            #img = x#resize(x.float(), (2048, 1024), mode = 'bilinear').int()#self.resize_shape_cloud)
            # print(img)
            data = dict()
            data['inputs'] = resize(x.float(), self.resize_shape_cloud, mode = 'bilinear').int()
            outputs =  self.cloud_model.test_step(data)
            foutputs_ = torch.stack([outputs[i].seg_logits.data.max(0).indices for i in range(len(x))])
            # fconfidence = torch.stack([outputs[i].seg_logits.data.softmax(0).max(0).values for i in range(len(x))])

        # img = resize(x, self.resize_shape_device, mode = 'bilinear') #self.resize_shape_cloud)
        data = dict()
        data['inputs'] = x
        outputs = self.device_model.test_step(data)
        outputs = torch.stack([outputs[i].seg_logits.data for i in range(len(x))])
        confidence = torch.max(outputs.softmax(1), dim=1).values
        # print(foutputs_.shape, outputs.shape, fconfidence.shape, confidence.shape
            #   )
        print(foutputs_.shape, outputs.shape)

        # print(fconfidence.mean(), confidence.mean())
        self.conf = confidence.mean()

        # outputs = resize(outputs, size = foutputs_.size()[1:])
        loss = F.cross_entropy(resize(outputs, self.resize_shape_cloud, mode='bilinear'), foutputs_).mean()
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
        # update_ema_variables(self.device_model_ema, self.device_model)

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
    parser.add_argument('--config_t', help='train config file path')
    parser.add_argument('--checkpoint_t', help='checkpoint file')
    parser.add_argument('--backbone', help='checkpoint file')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    print('CONFIG: ', args.config, args.config_t)
    print('CHECKPOINT: ', args.checkpoint, args.checkpoint_t)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint

    cfg_cm = Config.fromfile(args.config_t)
    cfg_cm.load_from = args.checkpoint_t

    register_all_modules()
    device = 'cuda:0'

    cloud_model = MODELS.build(cfg_cm.model)
    if args.backbone:
        checkpoint = load_checkpoint(cloud_model.backbone, args.backbone, map_location='cpu')
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