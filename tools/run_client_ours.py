# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import torch
import time
import threading
import ftp
import cv2

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

import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image
def visualize_segmentation(image, mask, name, class_colors=None):
    """
    Visualize the original image and its segmentation mask side by side.
    
    Args:
        image (np.ndarray): Original RGB image of shape (H, W, 3).
        mask (np.ndarray): Segmentation mask of shape (H, W), 
                           where each pixel is a class ID.
        class_colors (list or np.ndarray, optional): A list of RGB triplets 
                                                     for each class ID. 
                                                     If None, random colors are used.
    """
    # Check shapes
    assert len(image.shape) == 3 and image.shape[2] == 3, "image should be (H, W, 3)"
    assert len(mask.shape) == 2, "mask should be (H, W)"
    assert image.shape[:2] == mask.shape, "image and mask must have the same spatial dimensions"
    
    # If no custom class colors are provided, create a random colormap
    cmap = ListedColormap(np.array(class_colors) / 255.0)
    
    # Create a figure with two subplots
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Left: Original image
    img = Image.fromarray(image.astype(np.uint8))
    img.save(f'{name}_img.png')
    
    # Right: Segmentation mask
    # We use 'imshow' with a discrete colormap representing the class IDs
    color_mask = np.zeros_like(image)
    for class_id, color in enumerate(class_colors):
        # color should be [R, G, B]
        color_mask[mask == class_id] = color
    img = Image.fromarray(color_mask.astype(np.uint8))
    img.save(f'{name}_mask.png')

def main():
    # Network setup
    device_manager = ftp.DeviceManager(
        ul_host = '172.27.183.243', ul_port = 9999,
        dl_host = '172.27.183.242', dl_port = 9998,
        bpsspeed = 0
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


    if 'acdc' in args.config:
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
        condition_order = ['fog', 'night', 'rain', 'snow'] * 2
        resize_shape = (512, 1024)
        model_input_shape = (512, 1024)


    evaluator = IoUMetric()
    evaluator.dataset_meta = model.dataset_meta

    image_array = deque([])
    n_round = 1
    total_n_update = 0
    total_downlink_times = 0
    device_manager.image_send_limit = 8

    f = open('results/{}.txt'.format(args.save_path), 'w')

    total_uplink_time = 0
    total_downlink_time = 0
    start_time = time.time()
    for condition in condition_order:
        downlink_times = []

        start = time.time()
        n_update = 0

        for k, data in enumerate(condition_loader[condition]):
            # Save the image in array for future uplink.
            if len(image_array) < device_manager.image_send_limit:
                image_array.append(resize(data['inputs'][0].unsqueeze(0), resize_shape).cpu())
            else:
                image_array.popleft()
                image_array.append(resize(data['inputs'][0].unsqueeze(0), resize_shape).cpu())

            # Check the adaptation is in progress.
            if device_manager.is_in_progress:
                # Check the Downlink is finished.
                if device_manager.dl_is_finished:
                    # Update the model params.
                    updated, model = device_manager.update(model)
                    n_update += updated
                    downlink_times.append(device_manager.downlink_time)
                    # print('model is updated')
                    total_downlink_time += device_manager.downlink_time
                else:                    
                    # Go to the prediction process.
                    # device_manager.th.join()
                    pass
            else:
                # If not, start the uplink and wait and go to the prediction
                uplink_start = time.time()
                image_array = list(image_array)
                image_array = torch.cat(image_array, dim=0)
                device_manager.uplink(image_array)
                image_array = deque([])
                # device_manager.th.join()
                uplink_end = time.time()
                total_uplink_time += (uplink_end-uplink_start)

            with torch.no_grad():
                data_ = model.data_preprocessor(data)
                outputs = model.predict(resize(data_['inputs'], model_input_shape, mode='bilinear'))
                results = model.postprocess_result(outputs[0].seg_logits.data.unsqueeze(0), data['data_samples'])
                _data_samples = []
                for data_sample in results:
                        _data_samples.append(data_sample.to_dict())
                
                evaluator.process(data_samples=_data_samples, data_batch=data)


        total_n_update += n_update
        result = evaluator.evaluate(len(condition_loader[condition].dataset))
        print(len(condition_loader[condition].dataset))
        # print(result)
        # end = time.time()
        # print('{}-th round / condition: {}'.format(n_round, condition))
        # print('FPS: ', 1/((end-start)/len(condition_loader[condition])))
        # print('Number of update for this condition: {}'.format(n_update))
        # if len(downlink_times)>0:
        #     print('Average downlink time: {}'.format(sum(downlink_times)/len(downlink_times)))
        #     total_downlink_times += sum(downlink_times)/len(downlink_times)
        #     device_manager.join()
        # if 'acdc' in args.config and condition == 'snow':
        #     n_round += 1
        f.write('{}, '.format(result['mIoU']))
    end_time = time.time()

    # print('Mean number of update: {}'.format(total_n_update/len(condition_order)))
    # print('Total average downlink times: {}'.format(total_downlink_times/len(condition_order)))

    print('Total Time: ', end_time-start_time)
    print('Uplink Time: ', total_uplink_time)
    print('Downlink Time: ', total_downlink_time)

    f.close()

if __name__ == '__main__':
    main()