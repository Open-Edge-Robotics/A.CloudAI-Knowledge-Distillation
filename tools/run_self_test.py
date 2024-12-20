# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import torch

from mmseg.apis import init_model, show_result_pyplot
from mmseg.utils import register_all_modules, SampleList, dataset_aliases, get_classes, get_palette
from mmengine import mkdir_or_exist
from mmengine.config import Config, DictAction
from mmengine.runner import Runner, load_checkpoint
from mmengine.registry import MODELS, EVALUATOR, METRICS
# from mmseg.registry import DATASETS
from mmseg.models.utils import Upsample, resize
from mmseg.evaluation import IoUMetric

import time
import rein
# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument("--backbone", help="backbone checkpoint file", default="")
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
    mkdir_or_exist(cfg.work_dir)

    cfg.load_from = args.checkpoint
    if args.backbone:
        cfg.backbone = args.backbone


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
    if args.backbone:
        checkpoint = load_checkpoint(model.backbone, cfg.backbone, map_location='cpu')

    # print(checkpoint['state_dict'].keys())
    model.dataset_meta = {
                'classes': get_classes('cityscapes'),
                'palette': get_palette('cityscapes')
            }
    model.to(device)
    model.eval()

    testloader_clean = Runner.build_dataloader(cfg['test_dataloader_clean'])
    testloader_fog = Runner.build_dataloader(cfg['test_dataloader_fog'])
    testloader_night = Runner.build_dataloader(cfg['test_dataloader_night'])
    testloader_rain = Runner.build_dataloader(cfg['test_dataloader_rain'])
    testloader_snow = Runner.build_dataloader(cfg['test_dataloader_snow'])

    condition_loader = {'clean': testloader_clean,
                        'fog': testloader_fog,
                        'night': testloader_night,
                        'rain': testloader_rain,
                        'snow': testloader_snow}
    condition_order = ['clean', 'fog', 'night', 'rain', 'snow']

    evaluator = IoUMetric()
    evaluator.dataset_meta = model.dataset_meta

    resize_shape = (512, 1024)#model.test_cfg['crop_size']
    print(resize_shape)

    for condition in condition_order:
        print('Condition: ', condition)
        start = time.time()

        for k, data in enumerate(condition_loader[condition]):
            # print(data)
            # data = model.data_preprocessor(data)
#
            with torch.no_grad():
                # print(data['inputs'].shape, resize(data['inputs'], resize_shape).shape)
                # x = model.extract_feat(resize(data['inputs'], resize_shape))
                # print(data)
                # data['inputs'] = resize(data['inputs'][0].float(), resize_shape)
                outputs = model.test_step(data)
                # outputs =  model.decode_head.forward(x)
                # outputs = model(data['inputs'], mode='predict')
                outputs = outputs[0].seg_logits.data.unsqueeze(0) # For Rein 
                # print(outputs.shape)
                results = model.postprocess_result(outputs, data['data_samples'])
                _data_samples = []
                for data_sample in results:
                        _data_samples.append(data_sample.to_dict())
                evaluator.process(data_samples=_data_samples, data_batch=data)
        end = time.time()
        print(evaluator.evaluate(len(testloader_clean.dataset)))
        end = time.time()
        print((end-start)/len(condition_loader[condition]))

if __name__ == '__main__':
    main()
