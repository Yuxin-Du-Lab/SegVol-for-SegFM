# SegVol-for-SegFM
This repo is the SegFM version of SegVol for [SegFM](https://www.codabench.org/competitions/5263/).
SegFM is CVPR 2025: Foundation Models for Interactive 3D Biomedical Image Segmentation.

Origin repo of SegVol: https://github.com/BAAI-DCAI/SegVol
## Get Start
### Requirements
The [pytorch v1.13.1](https://pytorch.org/get-started/previous-versions/) (or a higher version) is needed first. Following install key requirements using commands:

```
pip install 'monai[all]==0.9.0'
pip install einops==0.6.1
pip install transformers==4.18.0
```

### Validation
1. Download our preliminary [**checkpoint**]() first and set the `ckpt_path` var in validation.py.

2. Download [SegFM](https://www.codabench.org/competitions/5263/) validation set and build **val_samples.json** file to index validation npz files like this:
```
[
"/path/to/xxx.npz",
"/path/to/yyy.npz"
]
```

3. Run:
```
python validation.py
```
