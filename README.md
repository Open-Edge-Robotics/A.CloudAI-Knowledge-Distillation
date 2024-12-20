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
