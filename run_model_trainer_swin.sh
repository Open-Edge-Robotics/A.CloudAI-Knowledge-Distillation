python host/run_model_trainer.py \
    configs/temp/segformer_mit-b1_acdc.py \
    ./segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth \
    --config_t configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_acdc.py \
    --checkpoint_t ./work_dirs/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024/iter_90000.pth \
    --cloud_model_shape 512
