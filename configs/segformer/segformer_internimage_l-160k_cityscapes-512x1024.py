_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/cityscapes_512x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
load_from = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/segformer_internimage_l_512x1024_80k_mapillary.pth'
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=160,
        depths=[5, 5, 22, 5],
        groups=[10, 20, 40, 80],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=None),
    decode_head=dict(num_classes=150, in_channels=[160, 320, 640, 1280]),
    test_cfg=dict(mode='whole'))


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=1, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
