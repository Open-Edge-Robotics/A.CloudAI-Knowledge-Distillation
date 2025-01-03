python host/run_model_trainer.py \
    configs/temp/segformer_mit-b1_acdc.py \
    ./segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth \
    --config_t configs/temp/segformer_mit-b5_acdc.py \
    --checkpoint_t ./segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth \
    --cloud_model_shape 512