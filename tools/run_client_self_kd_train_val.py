# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import torch
import torch.nn.functional as F
import time
import threading
import ftp

from collections import deque

from mmseg.apis import init_model
from mmseg.utils import register_all_modules, SampleList, dataset_aliases, get_classes, get_palette
from mmengine.config import Config, DictAction
from mmengine.runner import Runner, load_checkpoint
from mmengine.registry import MODELS, EVALUATOR, METRICS
# from mmseg.registry import DATASETS
from mmseg.models.utils import Upsample, resize
from mmseg.evaluation import IoUMetric
import rein

import logging
logging.basicConfig(level=logging.ERROR)


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
    parser.add_argument(
        '--backbone', type=str
    )
    parser.add_argument('--config_t', help='train config file path')
    parser.add_argument('--checkpoint_t', help='checkpoint file')

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

    cfg_cm = Config.fromfile(args.config_t)
    cfg_cm.load_from = args.checkpoint_t

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
    if args.backbone:
        checkpoint = load_checkpoint(cloud_model.backbone, args.backbone, map_location='cpu')
    checkpoint = load_checkpoint(cloud_model, cfg_cm.load_from, map_location='cpu')
    cloud_model.dataset_meta = {
                'classes': get_classes('cityscapes'),
                'palette': get_palette('cityscapes')
            }
    cloud_model.to(device)
    cloud_model.eval()

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
        trainloader_clear = Runner.build_dataloader(cfg['train_dataloader_clear'])
        trainloader_25 = Runner.build_dataloader(cfg['train_dataloader_25'])
        trainloader_50 = Runner.build_dataloader(cfg['train_dataloader_50'])
        trainloader_75 = Runner.build_dataloader(cfg['train_dataloader_75'])
        trainloader_100 = Runner.build_dataloader(cfg['train_dataloader_100'])
        trainloader_200 = Runner.build_dataloader(cfg['train_dataloader_200'])

        testloader_clear = Runner.build_dataloader(cfg['test_dataloader_clear'])
        testloader_25 = Runner.build_dataloader(cfg['test_dataloader_25'])
        testloader_50 = Runner.build_dataloader(cfg['test_dataloader_50'])
        testloader_75 = Runner.build_dataloader(cfg['test_dataloader_75'])
        testloader_100 = Runner.build_dataloader(cfg['test_dataloader_100'])
        testloader_200 = Runner.build_dataloader(cfg['test_dataloader_200'])



        condition_loader = {'clear': trainloader_clear,
                            '25': trainloader_25,
                            '50': trainloader_50,
                            '75': trainloader_75,
                            '100': trainloader_100,
                            '200': trainloader_200,

                            'clear_val': testloader_clear,
                            '25_val': testloader_25,
                            '50_val': testloader_50,
                            '75_val': testloader_75,
                            '100_val': testloader_100,
                            '200_val': testloader_200,
                            }
        condition_order = ['clear', '25', '50', '75', '100', '200', '100', '75', '50', '25', 'clear']

    elif 'acdc' in args.config:
        testloader_fog = Runner.build_dataloader(cfg['test_dataloader_fog'])
        testloader_night = Runner.build_dataloader(cfg['test_dataloader_night'])
        testloader_rain = Runner.build_dataloader(cfg['test_dataloader_rain'])
        testloader_snow = Runner.build_dataloader(cfg['test_dataloader_snow'])

        valloader_fog = Runner.build_dataloader(cfg['val_dataloader_fog'])
        valloader_night = Runner.build_dataloader(cfg['val_dataloader_night'])
        valloader_rain = Runner.build_dataloader(cfg['val_dataloader_rain'])
        valloader_snow = Runner.build_dataloader(cfg['val_dataloader_snow'])

        condition_loader = {
                            'fog': testloader_fog,
                            'night': testloader_night,
                            'rain': testloader_rain,
                            'snow': testloader_snow,

                            'fog_val': valloader_fog,
                            'night_val': valloader_night,
                            'rain_val': valloader_rain,
                            'snow_val': valloader_snow

                            }
        condition_order = ['fog', 'night', 'rain', 'snow'] * 10
    
    evaluator = IoUMetric()
    evaluator.dataset_meta = model.dataset_meta


    params = []
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad: #
            params.append(param)
            names.append(name)
    # print(names)
    print('model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('teacher: ', sum(p.numel() for p in cloud_model.parameters() if p.requires_grad))


    optimizer = torch.optim.Adam(params, lr=0.00008 / 8, betas=(0.9, 0.999))

    teacher_resize_shape = (1024, 1024)
    resize_shape = (512, 1024)
    n_round = 1
    total_n_update = 0
    f = open('results/{}.txt'.format(args.save_path), 'w')
    fv = open('results/{}_val.txt'.format(args.save_path), 'w')
    for condition in condition_order:
        downlink_times = []

        start = time.time()
        n_update = 0

        for i, data in enumerate(condition_loader[condition]):
            optimizer.zero_grad()
            # Inference the image
            # with torch.no_grad():
            data_ = model.data_preprocessor(data)
            outputs = model.predict(resize(data_['inputs'], resize_shape, mode='bilinear'))
            results = model.postprocess_result(outputs[0].seg_logits.data.unsqueeze(0), data_['data_samples'])

            if i % 5 == 0:
                with torch.no_grad():
                    data_ = cloud_model.data_preprocessor(data)
                    outputs_t = cloud_model.predict(resize(data_['inputs'], teacher_resize_shape, mode='bilinear'))
                    results_t = cloud_model.postprocess_result(outputs_t[0].seg_logits.data.unsqueeze(0), data_['data_samples'])

                loss = F.cross_entropy(results[0].seg_logits.data.unsqueeze(0), results_t[0].seg_logits.data.max(0).indices.unsqueeze(0)).mean()
                loss.backward()
                optimizer.step()

            _data_samples = []
            for data_sample in results:
                _data_samples.append(data_sample.to_dict())
            evaluator.process(data_samples=_data_samples, data_batch=data)
        total_n_update += n_update
        result = evaluator.evaluate(len(condition_loader[condition].dataset))
        print(result)
        print('{}-th round / condition: {}'.format(n_round, condition))
        f.write('{}, '.format(result['mIoU']))
        

        for data in condition_loader[condition+'_val']:
            with torch.no_grad():
                data_ = model.data_preprocessor(data)
                outputs = model.predict(resize(data_['inputs'], resize_shape, mode='bilinear'))
                results = model.postprocess_result(outputs[0].seg_logits.data.unsqueeze(0), data_['data_samples'])
            _data_samples = []
            for data_sample in results:
                _data_samples.append(data_sample.to_dict())
            evaluator.process(data_samples=_data_samples, data_batch=data)
        result = evaluator.evaluate(len(condition_loader[condition+'_val'].dataset))
        fv.write('{}, '.format(result['mIoU']))

        print(result)
        end = time.time()
        print('{}-th round / condition: {}'.format(n_round, condition))
        print('FPS: ', 1/((end-start)/len(condition_loader[condition])))
        if 'acdc' in args.config and condition == 'snow':
            n_round += 1
    f.close()
    fv.close()

if __name__ == '__main__':
    main()
