# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import torch
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
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def main():
    # Network setup
    device_manager = ftp.DeviceManager(
        ul_host = '192.168.0.1', ul_port = 9999,
        dl_host = '192.168.0.1', dl_port = 9998,
        )

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
        resize_shape = (512, 1024)


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
        resize_shape = (480, 960)
    
    evaluator = IoUMetric()
    evaluator.dataset_meta = model.dataset_meta

    image_array = deque([])
    n_round = 1
    total_n_update = 0
    total_downlink_times = 0

    f = open('results/{}.txt'.format(args.save_path), 'w')
    fv= open('results/{}_val.txt'.format(args.save_path), 'w')

    for condition in condition_order:
        downlink_times = []

        start = time.time()
        n_update = 0

        for data in condition_loader[condition]:
            # Save the image in array for future uplink.
            if len(image_array) < device_manager.image_send_limit:
                image_array.append(resize(data['inputs'][0].unsqueeze(0), resize_shape).cpu())
            else:
                image_array.popleft()
                image_array.append(resize(data['inputs'][0].unsqueeze(0), resize_shape).cpu())
            # data = model.data_preprocessor(data)

            # Check the adaptation is in progress.
            if device_manager.is_in_progress:
                # Check the Downlink is finished.
                # print(device_manager.is_in_progress, device_manager.dl_is_finished)
                if device_manager.dl_is_finished:
                    # Update the model params.
                    model = device_manager.update(model)
                    n_update += 1
                    downlink_times.append(device_manager.downlink_time)
                    # print('model is updated')
                else:                    
                    # Go to the prediction process.
                    pass
            else:
                # If not, start the uplink
                image_array = list(image_array)
                image_array = torch.cat(image_array, dim=0)
                device_manager.uplink(image_array)
                image_array = deque([])

            with torch.no_grad():
                data_ = model.data_preprocessor(data)
                outputs = model.predict(resize(data_['inputs'], resize_shape, mode='bilinear'))
                results = model.postprocess_result(outputs[0].seg_logits.data.unsqueeze(0), data_['data_samples'])
                _data_samples = []
                for data_sample in results:
                        _data_samples.append(data_sample.to_dict())
                
                evaluator.process(data_samples=_data_samples, data_batch=data)
        total_n_update += n_update
        result = evaluator.evaluate(len(condition_loader[condition].dataset))
        print(result)
        end = time.time()
        print('{}-th round / condition: {}'.format(n_round, condition))
        print('FPS: ', 1/((end-start)/len(condition_loader[condition])))
        print('Number of update for this condition: {}'.format(n_update))
        if len(downlink_times)>0:
            print('Average downlink time: {}'.format(sum(downlink_times)/len(downlink_times)))
            total_downlink_times += sum(downlink_times)/len(downlink_times)
            device_manager.join()
        if 'acdc' in args.config and condition == 'snow':
            n_round += 1
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
        print(result)
        fv.write('{}, '.format(result['mIoU']))

    print('Mean number of update: {}'.format(total_n_update/len(condition_order)))
    print('Total average downlink times: {}'.format(total_downlink_times/len(condition_order)))
    f.close()

if __name__ == '__main__':
    main()
