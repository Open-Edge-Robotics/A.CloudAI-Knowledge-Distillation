python tools/run_host_full.py \
    configs/temp/deeplabv3plus_r18-d8_4xb2-80k_cityscapes-512x1024_acdc.py \
    ./deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth \
    --config_t configs/dinov2_citys2acdc/rein_dinov2l_mask2former_1024x1024_bs4x2.py \
    --checkpoint_t ./work_dirs/rein_dinov2l_mask2former_1024x1024_bs4x2/iter_40000.pth \
    --backbone ./checkpoints/dinov2l_converted_1024x1024.pth