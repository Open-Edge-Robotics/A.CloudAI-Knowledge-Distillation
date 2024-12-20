python tools/run_client_self_kd_train_val.py \
    configs/temp/segformer_mit-b1_adaptformer_storm.py \
    ./segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth \
    --config_t configs/dinov2_citys2acdc/rein_dinov2s_mask2former_1024x1024_bs4x2.py \
    --checkpoint_t ./work_dirs/rein_dinov2s_mask2former_1024x1024_bs4x2/iter_40000.pth \
    --backbone ./checkpoints/dinov2s_converted_1024x1024.pth