# checkpoint = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'  # noqa
checkpoint = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'
# model settings
# backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(
        type='DinoVisionTransformer',
        img_size=(518, 518),
        patch_size=14,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
    ),
    decode_head=dict(
        type='LinearHeadMS',
    ),
    test_cfg=dict(mode='slide', crop_size=(518, 518), stride=(480, 480)),
)
