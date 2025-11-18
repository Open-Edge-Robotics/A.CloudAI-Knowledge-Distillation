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

import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image

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
    parser.add_argument(
        '--save_path', type=str, default='None', help='Test time augmentation')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

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

    # testloader_clean = Runner.build_dataloader(cfg['test_dataloader_clean'])
    # testloader_fog = Runner.build_dataloader(cfg['test_dataloader_fog'])
    # testloader_night = Runner.build_dataloader(cfg['test_dataloader_night'])
    # testloader_rain = Runner.build_dataloader(cfg['test_dataloader_rain'])
    # testloader_snow = Runner.build_dataloader(cfg['test_dataloader_snow'])

    # condition_loader = {'clean': testloader_clean,
    #                     'fog': testloader_fog,
    #                     'night': testloader_night,
    #                     'rain': testloader_rain,
    #                     'snow': testloader_snow}
    # condition_order = ['fog', 'night', 'rain', 'snow']

    if 'shift' in args.config:
        # video = [x for x in os.listdir('./data/shift/video') if not x.endswith('.py')]
        video = ['d11b-8666', '6c4c-ec9b', '1ee5-e8db', 'a30f-b210', '3c95-8ad5', 'f316-dc0b', '2ecd-cce4', '3b8a-b336', '7233-b8c2', 'e5f3-bbdc', 'a9cb-54e3', 'af55-3500', '1654-0260', '520d-0b70', 'b414-5936', '7900-e2cd', '7048-15e1', '8def-85dc', '98d1-af8d', '4a0d-4564', '146a-d226', '337e-11d0', '7fbc-c771', '364b-0733', '4f6e-e7e1', 'deb7-6032']

        test_loader = Runner.build_dataloader(cfg['test_dataloader'])
        condition_loader = {}
        condition_order = []

        for v in video:
            test_loader = Runner.build_dataloader(cfg[f'test_dataloader_{v[:4]}'])
            condition_loader[v] = test_loader
            condition_order.append(v)
    
    print(condition_order)


    evaluator = IoUMetric()
    evaluator.dataset_meta = model.dataset_meta

    resize_shape = (512, 1024) #model.test_cfg['crop_size']
    print(resize_shape)

    f = open('results/SHIFT_{}.txt'.format(args.save_path), 'w')
    for condition in condition_order:
        print('Condition: ', condition)
        start = time.time()

        for k, data in enumerate(condition_loader[condition]):
            # print(data)
            # data = model.data_preprocessor(data)
#
            with torch.no_grad():      
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

                # if k < 5:
                #     # print(data['inputs'][0].shape, outputs.shape)
                #     # print(results[0].pred_sem_seg.data.shape)
                #     # print(results)
                    # pred = results[0].pred_sem_seg.data.squeeze(0).cpu().numpy()
                    # print(pred.shape)
                #     gt = results[0].gt_sem_seg.data.squeeze(0).cpu().numpy()

                    # img = np.array(Image.open(results[0].img_path).convert('RGB').resize([1280, 800]))
                #     # print(pred.shape, img.shape)
                    # visualize_segmentation(img, pred, f'{condition}_{k}_b1', class_colors=model.dataset_meta['palette'])
                #     visualize_segmentation(img, gt, f'{condition}_{k}_gt', class_colors=model.dataset_meta['palette'])
        print(k)
        result = evaluator.evaluate(len(condition_loader[condition]))
        end = time.time()
        print(result)
        f.write('{} '.format(result['mIoU']))
        print((end-start)/len(condition_loader[condition]))
    f.close()

if __name__ == '__main__':
    main()