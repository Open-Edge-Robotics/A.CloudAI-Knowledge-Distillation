dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_crop_size = (1024, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2049, 1025),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
cotta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(540, 960), keep_ratio=True),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ]),

]
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 1024), keep_ratio=True),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
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

train_dataloader_clear = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'),
        pipeline=test_pipeline))

train_dataloader_25 = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/weather_datasets/weather_cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/train/rain/25mm/rainy_image', seg_map_path='../../gtFine/train'),
        pipeline=test_pipeline)
)
train_dataloader_50 = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/weather_datasets/weather_cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/train/rain/50mm/rainy_image', seg_map_path='../../gtFine/train'),
        pipeline=test_pipeline)
)
train_dataloader_75 = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/weather_datasets/weather_cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/train/rain/75mm/rainy_image', seg_map_path='../../gtFine/train'),
        pipeline=test_pipeline)
)
train_dataloader_100 = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/weather_datasets/weather_cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/train/rain/100mm/rainy_image', seg_map_path='../../gtFine/train'),
        pipeline=test_pipeline)
)
train_dataloader_200 = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/weather_datasets/weather_cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/train/rain/200mm/rainy_image', seg_map_path='../../gtFine/train'),
        pipeline=test_pipeline)
)


test_dataloader_clear = dict(
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

test_dataloader_25 = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/weather_datasets/weather_cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/val/rain/25mm/rainy_image', seg_map_path='../../gtFine/val'),
        pipeline=test_pipeline)
)
test_dataloader_50 = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/weather_datasets/weather_cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/val/rain/50mm/rainy_image', seg_map_path='../../gtFine/val'),
        pipeline=test_pipeline)
)
test_dataloader_75 = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/weather_datasets/weather_cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/val/rain/75mm/rainy_image', seg_map_path='../../gtFine/val'),
        pipeline=test_pipeline)
)
test_dataloader_100 = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/weather_datasets/weather_cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/val/rain/100mm/rainy_image', seg_map_path='../../gtFine/val'),
        pipeline=test_pipeline)
)
test_dataloader_200 = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/weather_datasets/weather_cityscapes/',
        data_prefix=dict(
            img_path='leftImg8bit/val/rain/200mm/rainy_image', seg_map_path='../../gtFine/val'),
        pipeline=test_pipeline)
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator