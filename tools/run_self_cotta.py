# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import torch
import time
import threading
import ftp

from collections import deque
from torchvision.transforms.functional import hflip
from mmseg.apis import init_model
from mmseg.utils import register_all_modules, SampleList, dataset_aliases, get_classes, get_palette
from mmengine.config import Config, DictAction
from mmengine.runner import Runner, load_checkpoint
from mmengine.registry import MODELS, EVALUATOR, METRICS
# from mmseg.registry import DATASETS
from mmseg.models.utils import Upsample, resize
from mmseg.evaluation import IoUMetric
from copy import deepcopy

import logging
logging.basicConfig(level=logging.ERROR)

def create_ema_model(model):
    ema_model = deepcopy(model)#get_model(args.model)(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
    #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher=0.999, iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    if True:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def cross_entropy(x, x_ema):# -> torch.Tensor:
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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
    parser.add_argument(
        '--save_path', type=str, help='text file name for saving evaluation'
    )
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    args = parse_args()

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

    model = MODELS.build(cfg.model)
    checkpoint = load_checkpoint(model, cfg.load_from, map_location='cpu')
    model.dataset_meta = {
                'classes': get_classes('cityscapes'),
                'palette': get_palette('cityscapes')
            }
    
    model.to(device)
    model.eval()

    print(args.config)
    if 'storm' in args.config:
        cfg.test_dataloader_clear.dataset.pipeline = cfg.cotta_pipeline
        cfg.test_dataloader_25.dataset.pipeline = cfg.cotta_pipeline
        cfg.test_dataloader_50.dataset.pipeline = cfg.cotta_pipeline
        cfg.test_dataloader_75.dataset.pipeline = cfg.cotta_pipeline
        cfg.test_dataloader_100.dataset.pipeline = cfg.cotta_pipeline
        cfg.test_dataloader_200.dataset.pipeline = cfg.cotta_pipeline

        testloader_clear = Runner.build_dataloader(cfg['test_dataloader_clear'])
        testloader_25 = Runner.build_dataloader(cfg['test_dataloader_25'])
        testloader_50 = Runner.build_dataloader(cfg['test_dataloader_50'])
        testloader_75 = Runner.build_dataloader(cfg['test_dataloader_75'])
        testloader_100 = Runner.build_dataloader(cfg['test_dataloader_100'])
        testloader_200 = Runner.build_dataloader(cfg['test_dataloader_200'])


        condition_loader = {'clear': testloader_clear,
                            '25': testloader_25,
                            '50': testloader_50,
                            '75': testloader_75,
                            '100': testloader_100,
                            '200': testloader_200,
                            }
        condition_order = ['clear', '25', '50', '75', '100', '200', '100', '75', '50', '25', 'clear']

    elif 'acdc' in args.config:
        cfg.test_dataloader_fog.dataset.pipeline = cfg.cotta_pipeline
        cfg.test_dataloader_night.dataset.pipeline = cfg.cotta_pipeline
        cfg.test_dataloader_rain.dataset.pipeline = cfg.cotta_pipeline
        cfg.test_dataloader_snow.dataset.pipeline = cfg.cotta_pipeline

        testloader_fog = Runner.build_dataloader(cfg['test_dataloader_fog'])
        testloader_night = Runner.build_dataloader(cfg['test_dataloader_night'])
        testloader_rain = Runner.build_dataloader(cfg['test_dataloader_rain'])
        testloader_snow = Runner.build_dataloader(cfg['test_dataloader_snow'])

        condition_loader = {
                            'fog': testloader_fog,
                            'night': testloader_night,
                            'rain': testloader_rain,
                            'snow': testloader_snow
                            }
        condition_order = ['fog', 'night', 'rain', 'snow'] * 100
        #Optimizer setting (Norm parameters are selected for updating)

    params = []
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.requires_grad and "backbone" in name:
                params.append(param)
                names.append(name)
    print(names)
    optimizer = torch.optim.Adam(params, lr=0.00006/8, betas=(0.9, 0.999))

    # Create ema model and anchor model
    anchor = deepcopy(model.state_dict())
    anchor_model = deepcopy(model)
    ema_model = create_ema_model(model)

    model.eval()
    anchor_model.eval()
    ema_model.eval()

    evaluator = IoUMetric()
    evaluator.dataset_meta = model.dataset_meta

    resize_shape = (480, 960)
    image_array = deque([])
    n_round = 1
    total_n_update = 0
    total_downlink_times = 0

    f = open('results/{}.txt'.format(args.save_path), 'w')
    for condition in condition_order:
        downlink_times = []

        start = time.time()
        n_update = 0

        for data in condition_loader[condition]:
            ori_idx = 4
            ori_inputs = data['inputs'][ori_idx][0]
            ori_data_samples = data['data_samples'][ori_idx]
            ori_data = {'inputs': ori_inputs.unsqueeze(0), 'data_samples': ori_data_samples}
            ori_data = model.data_preprocessor(ori_data)

            with torch.no_grad():
                result_ema = ema_model(**ori_data)
                result_ema = ema_model.postprocess_result(result_ema, ori_data['data_samples'])

                preds = result_ema[0].seg_logits

                result_anc = anchor_model(**ori_data)                    
                result_anc = resize(result_anc, ori_data['data_samples'][0].ori_shape, mode='bilinear')
                result_anc = result_anc.softmax(1).max(1).values
                mask = (result_anc[0] > 0.69).type(torch.int64).unsqueeze(0)

                tta_results = []
                for j in range(len(data['inputs'])):
                    data_ = {'inputs': data['inputs'][j][0].unsqueeze(0), 
                                'data_samples': data['data_samples'][j]}
                    data_ = ema_model.data_preprocessor(data_)
                    result = ema_model(**data_)
                    result = resize(result, ori_data['data_samples'][0].ori_shape, mode='bilinear')
                    if j % 2 == 1: # Flip the output for the given flipped images
                        result = hflip(result)
                    tta_results.append(result)
                                        
                tta_results = torch.cat(tta_results).mean(0, keepdim=True)

            pseudo_label = (mask*preds.data.unsqueeze(0) + (1.-mask)*tta_results).max(1).indices

            # print(ori_data['data_samples'])
            temp = ori_data['data_samples'][0].gt_sem_seg.data #= pseudo_label
            ori_data['data_samples'][0].gt_sem_seg.data = pseudo_label
            loss = model.loss(**ori_data)
            loss['decode.loss_ce'].backward()
            optimizer.step()
            optimizer.zero_grad()

            ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=0.999)
            for nm, m  in model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<0.01).float().cuda() 
                        with torch.no_grad():
                            p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)

            ori_data['data_samples'][0].gt_sem_seg.data = temp
            _data_samples = []
            for data_sample in result_ema:
                data_sample.pred_sem_seg.data = pseudo_label
                _data_samples.append(data_sample.to_dict())
            evaluator.process(data_samples=_data_samples, data_batch=ori_data)
        result = evaluator.evaluate(len(condition_loader[condition].dataset))
        print(result)
        end = time.time()
        print('{}-th round / condition: {}'.format(n_round, condition))
        print('FPS: ', 1/((end-start)/len(condition_loader[condition])))
        print('Number of update for this condition: {}'.format(n_update))

        if 'acdc' in args.config and condition == 'snow':
            n_round += 1
        f.write('{}, '.format(result['mIoU']))
    print('Mean number of update: {}'.format(total_n_update/len(condition_order)))
    print('Total average downlink times: {}'.format(total_downlink_times/len(condition_order)))
    f.close()

if __name__ == '__main__':
    main()
