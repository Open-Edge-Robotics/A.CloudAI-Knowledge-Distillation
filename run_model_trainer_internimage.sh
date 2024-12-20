python host/run_model_trainer.py \
    configs/temp/segformer_mit-b1_acdc.py \
    ./segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth \
    --config_t configs/segformer/segformer_internimage_l-160k_cityscapes-512x1024_acdc.py \
    --checkpoint_t ./segformer_internimage_l_512x1024_160k_mapillary2cityscapes.pth
    --cloud_model_shape 512
