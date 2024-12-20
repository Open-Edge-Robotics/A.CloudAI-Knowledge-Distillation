import cv2
import numpy as np
import os
import os.path as osp
import torch
import argparse

from mmseg.apis import init_model
from mmseg.utils import register_all_modules, SampleList, dataset_aliases, get_classes, get_palette
from mmengine.config import Config, DictAction
from mmengine.runner import Runner, load_checkpoint
from mmengine.registry import MODELS, EVALUATOR, METRICS
from mmseg.models.utils import Upsample, resize
from mmseg.evaluation import IoUMetric

from io import BytesIO
from pyftpdlib.authorizers import DummyAuthorizer
from pyftpdlib.handlers import FTPHandler, ThrottledDTPHandler
from pyftpdlib.servers import FTPServer

from ftplib import FTP

def symmetric_cross_entropy(x, x_ema):# -> torch.Tensor:
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)



class MyHandler(FTPHandler):
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

        self.ftp = FTP()
        self.ftp.connect(self.remote_ip, 9998)
        self.ftp.login(self.username, self.password)
        self.ftp.storbinary(f"STOR {'temp'}", dict_bytes_io)
        self.ftp.close()

    def on_connect(self):
        print(f"Connected to {self.remote_ip}")

    def on_file_received(self, file_path):
        with self.fs.open(file_path, 'rb') as file_bytes_io:
            array = np.load(file_bytes_io)
        print("Received NumPy array with shape:", array.shape)

        self.optimizer.zero_grad()
        x = torch.tensor(array, device = 'cuda')

        with torch.no_grad():
            feat = self.cloud_model.extract_feat(resize(x, self.resize_shape_cm))
            foutputs =  self.cloud_model.decode_head.forward(feat)
            foutputs = resize(foutputs, size=self.resize_shape_em)
            # print(outputs.shape)
        feat = self.edge_model.extract_feat(resize(x, self.resize_shape_em))
        outputs = self.edge_model.decode_head.forward(feat)
        outputs = resize(outputs, size=self.resize_shape_em)

        loss = symmetric_cross_entropy(outputs, foutputs).mean()
        loss.backward()
        self.optimizer.step()
        print("Edge model is adjusted by the cloud model.")
        self.send_delta_params()
        print("Send delta params back to the client.")

    def on_disconnect(self):
        print(f"Disconnected from {self.remote_ip}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('config_cm', help='train config file path')
    parser.add_argument('checkpoint_cm', help='checkpoint file')

    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--out',
        type=str,
        help='The directory to save output prediction for offline evaluation')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def setup_host(cloud_model, edge_model):
    # Create FTP server
    HOST = '210.125.85.243' # Replace with your server's IP or hostname
    PORT = 9999 # 

    # Set up FTP server authorizer
    authorizer = DummyAuthorizer()
    authorizer.add_user("coa", "sdfsdf", "./", perm="elradfmw")

    # Create FTP handler with the custom handler class
    dtp_handler = ThrottledDTPHandler
    dtp_handler.read_limit = 1024000 * 50  # 1Mb / sec = 1,000 Kb/sec (1000 * 1024)
    dtp_handler.write_limit = 1024000 * 50  # 1,000 Kb/sec (1000 * 1024)

    handler = MyHandler
    handler.authorizer = authorizer
    handler.dtp_handler = dtp_handler
    handler.cloud_model = cloud_model
    handler.edge_model = edge_model
    handler.resize_shape_cm = cloud_model.test_cfg['crop_size']
    handler.resize_shape_em = edge_model.test_cfg['crop_size']

    #Optimizer setting (Norm parameters are selected for updating)
    params = []
    for name, param in edge_model.named_parameters():
        if param.requires_grad: # and ("norm" in name or "bn" in name):
            params.append(param)
    optimizer = torch.optim.Adam(params, lr=0.00006/8, betas=(0.9, 0.999))
    handler.optimizer = optimizer

    server = FTPServer((HOST, PORT), handler)
    
    print("FTP Server running on {}:{}".format(HOST, PORT))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("FTP Server shutting down.")
        server.close_all()

def main():
    args = parse_args()
    print('CONFIG: ', args.config, args.config_cm)
    print('CHECKPOINT: ', args.checkpoint, args.checkpoint_cm)

    
    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    cfg_cm = Config.fromfile(args.config_cm)
    cfg_cm.load_from = args.checkpoint_cm

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True

    register_all_modules()
    device = 'cuda:0'

    cloud_model = MODELS.build(cfg_cm.model)
    print('aa', cfg_cm.load_from)
    checkpoint = load_checkpoint(cloud_model, cfg_cm.load_from, map_location='cpu')
    cloud_model.dataset_meta = {
                'classes': get_classes('cityscapes'),
                'palette': get_palette('cityscapes')
            }
    cloud_model.to(device)
    cloud_model.eval()

    edge_model = MODELS.build(cfg.model)
    checkpoint = load_checkpoint(edge_model, cfg.load_from, map_location='cpu')
    edge_model.dataset_meta = {
                'classes': get_classes('cityscapes'),
                'palette': get_palette('cityscapes')
            }
    
    edge_model.to(device)
    edge_model.eval()


    setup_host(cloud_model = cloud_model, edge_model= edge_model)

if __name__ == '__main__':
    main()