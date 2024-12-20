dataset_type = 'ACDCDataset'
data_root = 'data/data_acdc/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (490, 980)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2049, 1025),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes',
        img_dir='train',
        ann_dir='gtFine/train',
        pipeline=train_pipeline),
        
    val=dict(pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgb_anon/fog/train',
        ann_dir='gt/fog/train',
        pipeline=test_pipeline),
    test1=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgb_anon/night/train',
            ann_dir='gt/night/train',
        pipeline=test_pipeline),
    test2=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgb_anon/rain/train',
        ann_dir='gt/rain/train',
        pipeline=test_pipeline),
    test3=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='rgb_anon/snow/train',
        ann_dir='gt/snow/train',
        pipeline=test_pipeline))


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        pipeline=train_pipeline))


val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            # img_path='rgb_anon/fog/train', seg_map_path='gt/fog/train'),
            # img_path='rgb_anon/night/train', seg_map_path='gt/night/train'),
            # img_path='rgb_anon/rain/train', seg_map_path='gt/rain/train'),
            img_path='rgb_anon/snow/train', seg_map_path='gt/snow/train'),
        pipeline=test_pipeline))

test_dataloader_clean = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'),
        pipeline=test_pipeline)
)


test_dataloader_fog = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='rgb_anon/fog/train', seg_map_path='gt/fog/train'),
        pipeline=test_pipeline))

test_dataloader_night = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='rgb_anon/night/train', seg_map_path='gt/night/train'),
        pipeline=test_pipeline))

test_dataloader_rain = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='rgb_anon/rain/train', seg_map_path='gt/rain/train'),
        pipeline=test_pipeline))

test_dataloader_snow = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='rgb_anon/snow/train', seg_map_path='gt/snow/train'),
        pipeline=test_pipeline))


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator