# Foundation Model-based Cloud-Device Collaborative Online Adaptation with Effective Linking

---
### Environment setting
- install conda env
```
conda env create -f env.yaml
```

- install pytorch (e.g., we use pytorch version of 2.3.1
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


### Run
```
# Cloud Computer Terminal 1
sh run_cloud_manager.sh

# Cloud Computer Terminal 2
sh run_model_trainer.sh
```
---
```
# Device Computer Terminal 1
sh run_client_segb1_project.sh # For baseline

sh run_client_segb1_adapt_project.sh # For adaptformer
```