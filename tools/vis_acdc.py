# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import torch
import torch.nn as nn

from mmseg.apis import init_model
from mmseg.utils import register_all_modules, SampleList, dataset_aliases, get_classes, get_palette
from mmengine.config import Config, DictAction
from mmengine.runner import Runner, load_checkpoint
from mmengine.registry import MODELS, EVALUATOR, METRICS
# from mmseg.registry import DATASETS
from mmseg.models.utils import Upsample, resize
from mmseg.evaluation import IoUMetric

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

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

    # state_dict = model.state_dict()
    # for k, key in enumerate(state_dict):
    #     print(key, state_dict[key].shape)
    # exit()

    fmodel = MODELS.build(Config.fromfile('configs/temp/vit-g_acdc.py').model)
    checkpoint = load_checkpoint(fmodel, 'work_dirs/vit-g/iter_160000.pth', map_location='cpu')
    fmodel.dataset_meta = {
                'classes': get_classes('cityscapes'),
                'palette': get_palette('cityscapes')
            }
    fmodel.to(device)
    fmodel.eval()


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

    resize_shape = model.test_cfg['crop_size']
    f_resize_shape = fmodel.test_cfg['crop_size']

    with torch.no_grad():
        entropy_summer = nn.Conv2d(1, 1, [60, 60], [60, 60], bias=False).cuda()
        entropy_summer.weight.fill_(1)
    print(entropy_summer.weight)
    print(model)
    # print(resize_shape)
    for condition in condition_order:
        print('Condition: ', condition)

        for data in condition_loader[condition]:
            data = model.data_preprocessor(data)

            with torch.no_grad():
                x = model.extract_feat(resize(data['inputs'], resize_shape))
                x_original = resize(data['inputs'], (1080, 1920))
                x_low = resize(data['inputs'], (90, 160))
                x_resized = resize(x_low, (1080, 1920), mode='bilinear')

                outputs =  model.decode_head.forward(x)

                results = model.postprocess_result(outputs, data['data_samples'])
                logits = results[0].seg_logits.data.unsqueeze(0)
                entropy = softmax_entropy(logits)
                entropy_grid_sum = entropy_summer(entropy)
                entropy_grid_sum = (entropy_grid_sum-entropy_grid_sum.min())/entropy_grid_sum.max()
                entropy_grid_sum = entropy_grid_sum>0.5
                # print(entropy_grid_sum.sum())


                for x in range(entropy_grid_sum.size(2)):
                    for y in range(entropy_grid_sum.size(1)):
                        # print(x, y, entropy_grid_sum[0, y, x])
                        if entropy_grid_sum[0, y, x] == True:
                            x_resized[0, :, y*60: y*60+60, x*60: x*60+60] = x_original[0, :, y*60: y*60+60, x*60: x*60+60]

                fx = fmodel.extract_feat(resize(x_resized, f_resize_shape))
                foutputs =  fmodel.decode_head.forward(fx)
                results = fmodel.postprocess_result(foutputs, data['data_samples'])

                break
                # img = x_resized[0].permute(1, 2, 0).cpu()
                # img = (x_resized[0].cpu() * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))  + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                # img = img.permute(1,2,0)
                            
                # img = (np.array(img * 255)).astype(np.uint8)
                # img = Image.fromarray(img)
                # img.save('test.png')

                # break
                _data_samples = []
                for data_sample in results:
                        _data_samples.append(data_sample.to_dict())

                # plt.imshow(entropy_grid_sum.permute(1,2,0).cpu().numpy())
                # plt.savefig('test.png')
                evaluator.process(data_samples=_data_samples, data_batch=data)

        print(evaluator.evaluate(len(condition_loader[condition].dataset)))

    

if __name__ == '__main__':
    main()
