# A.CloudAI-Knowledge-Distillation
---
Knowledge distillation from Cloud/Edge AI model to Device AI model.

---
### Environment setting
- install conda env
```
conda env create -f env.yaml
```

- install pytorch (e.g., we use pytorch version of 2.3.1)
``` 
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

- install mmsegmentation ([reference](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/get_started.md#installation))
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

pip install -v -e .
```
---

### How to run
- activate conda env
```
conda activate cloud
```

- train the model (e.g., segformer-b1 (Device AI model))
```
python tools/train.py configs/segformer/segformer_mit-b1_8xb1-160k_cityscapes-1024x1024.py
```
- train the model (e.g., DINOv2-Large-Rein (Cloud AI model))
```
python tools/train.py configs/dinov2_citys2acdc/rein_dinov2l_mask2former_1024x1024_bs4x2.py
```

- run host manager in cloud
```
sh run_cloud_manager.sh
sh run_model_trainer.sh
```

- run client in device
```
python tools/run_client.py
```