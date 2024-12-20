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

def generate_pair_wise_map(features):
    m_batchsize, C, width, height = features.size()

    # self attn map
    features = features.view(m_batchsize, -1, width*height)
    features = features/torch.norm(features, p=2, dim=1, keepdim=True)

    attn_map = torch.bmm(features.permute(0, 2, 1), features)

    return attn_map

def device_model_forward(model, img):
    data = dict()
    data['inputs'] = img
    data = model.data_preprocessor(data, False)
    batch_img_metas = [
        dict(
            ori_shape=data['inputs'].shape[2:],
            img_shape=data['inputs'].shape[2:],
            pad_shape=data['inputs'].shape[2:],
            padding_size=[0, 0, 0, 0])
    ] * data['inputs'].shape[0]

    x, features = model.extract_feat(data['inputs'])
    
    outs = []
    for i, features in enumerate(features):
        f = model.backbone.adapter_decoder_conv[i](features)
        f = resize(f, (128, 256), mode = 'bilinear')
        outs.append(f)
    outs = torch.cat(outs, dim=1)
    outs = model.backbone.adapter_fusion_conv(outs)
    # print(outs.shape)

    seg_logits = model.decode_head.predict(x, batch_img_metas,
                                                model.test_cfg)
    return outs, seg_logits

def cloud_model_forward(model, img):
    data = dict()
    data['inputs'] = img
    data = model.data_preprocessor(data, False)
    batch_img_metas = [
        dict(
            ori_shape=data['inputs'].shape[2:],
            img_shape=data['inputs'].shape[2:],
            pad_shape=data['inputs'].shape[2:],
            padding_size=[0, 0, 0, 0])
    ] * data['inputs'].shape[0]

    x = model.extract_feat(data['inputs'])
    seg_logits = model.decode_head.predict(x, batch_img_metas,
                                                model.test_cfg)
    return x[0], seg_logits.max(1).indices

def calculate_fkd(cloud_features, device_features):
    n_batch, n_c, width, height = cloud_features[-1].size()
    device_features = resize(device_features, (width, height), mode = 'bilinear')    
    cloud_features = torch.cat([resize(x, (width, height), mode = 'bilinear') for x in cloud_features], dim=0)
    # print(cloud_features.shape)
    cloud_features = cloud_features.mean(0, keepdim=True)
    # width, height = 128, 128

    fs_loss = 1- F.cosine_similarity(device_features, cloud_features)
    fs_loss = fs_loss.mean()
    # fs_loss = F.mse_loss(device_features, cloud_features[-1])

    device_correltation = generate_pair_wise_map(device_features)
    cloud_correltation = generate_pair_wise_map(cloud_features)

    cor_loss = F.mse_loss(device_correltation, cloud_correltation)
    return fs_loss + cor_loss
    
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
    model.backbone.is_cloud = True
    model.backbone.set_cloud()

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

    elif 'acdc' in args.config:
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
        condition_order = ['fog', 'night', 'rain', 'snow'] * 10
    
    evaluator = IoUMetric()
    evaluator.dataset_meta = model.dataset_meta


    params = []
    names = []
    for name, param in model.named_parameters():
        if param.requires_grad and 'adapt' in name: #
            params.append(param)
            names.append(name)
    # print(names)
    print('model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('model_adapter: ', sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and 'adapt' in n))
    print('teacher: ', sum(p.numel() for p in cloud_model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(params, lr=0.00008 / 8, betas=(0.9, 0.999))

    teacher_resize_shape = (1024, 1024)
    resize_shape = (512, 1024)
    n_round = 1
    total_n_update = 0
    f = open('results/{}.txt'.format(args.save_path), 'w')

    train_loader = Runner.build_dataloader(cfg['train_dataloader'])
    for i, data in enumerate(train_loader):
        print('\r', i, end='')
        optimizer.zero_grad()
        img = data['inputs'][0].unsqueeze(0)

        # Inference the image
        with torch.no_grad():
            # data_ = cloud_model.data_preprocessor(data)
            # outputs_t = cloud_model.predict(resize(data_['inputs'], teacher_resize_shape, mode='bilinear'))
            # results_t = cloud_model.postprocess_result(outputs_t[0].seg_logits.data.unsqueeze(0), data_['data_samples'])
            features_t, outputs_t = cloud_model_forward(cloud_model, resize(img, teacher_resize_shape, mode='bilinear'))
            # print(len(features_t))
            # for f in features_t:
            #     print(f.shape)
        
        features, outputs = device_model_forward(model, resize(img, resize_shape, mode='bilinear'))
        # print(len(features))
        # for f in features:
        #     print(f.shape)


        # data_ = model.data_preprocessor(data)
        # outputs = model.predict(resize(data_['inputs'], resize_shape, mode='bilinear'))
        #results = model.postprocess_result(outputs.unsqueeze(0), data['data_samples'])

        celoss = F.cross_entropy(resize(outputs, teacher_resize_shape, mode='bilinear'), outputs_t).mean()
        fkd_loss = calculate_fkd(features_t, features)

        loss = celoss + fkd_loss
        loss.backward()
        optimizer.step()

    for condition in condition_order:
        downlink_times = []

        start = time.time()
        n_update = 0

        for i, data in enumerate(condition_loader[condition]):
            optimizer.zero_grad()
            # Inference the image
            # with torch.no_grad():
            # data_ = model.data_preprocessor(data)
            # outputs = model.predict(resize(data_['inputs'], resize_shape, mode='bilinear'))
            # results = model.postprocess_result(outputs[0].seg_logits.data.unsqueeze(0), data_['data_samples'])
            # print(data['inputs'][0].shape)
            img = data['inputs'][0].unsqueeze(0)
            features, outputs = device_model_forward(model, resize(img, resize_shape, mode='bilinear'))
            # print(outputs.shape)
            results = model.postprocess_result(outputs, data['data_samples'])


            if i % 5 == 0:
                with torch.no_grad():
                    # data_ = cloud_model.data_preprocessor(data)
                    # outputs_t = cloud_model.predict(resize(data_['inputs'], teacher_resize_shape, mode='bilinear'))
                    # results_t = cloud_model.postprocess_result(outputs_t[0].seg_logits.data.unsqueeze(0), data_['data_samples'])
                    features_t, outputs_t = cloud_model_forward(cloud_model, resize(img, teacher_resize_shape, mode='bilinear'))
                celoss = F.cross_entropy(resize(outputs, teacher_resize_shape, mode='bilinear'), outputs_t).mean()
                fkd_loss = calculate_fkd(features_t, features)

                loss = celoss + fkd_loss
                loss.backward()
                optimizer.step()

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
        if 'acdc' in args.config and condition == 'snow':
            n_round += 1
        f.write('{}, '.format(result['mIoU']))
    f.close()

if __name__ == '__main__':
    main()
