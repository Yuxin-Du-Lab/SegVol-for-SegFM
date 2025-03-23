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

### Validation with source code
1. Download our preliminary [**checkpoint** (2000 epochs ckpt)](https://drive.google.com/file/d/1dgM5slT5kDV3D_6k_vGpGqU5yB1nTwCL/view?usp=drive_link) first and set the `ckpt_path` var in validation.py. (Trained on 10% dataset and support box prompt only)
   

3. Download [SegFM](https://www.codabench.org/competitions/5263/) validation set and build **val_samples.json** file to index validation npz files like this:
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

### Train with source code
1. **Train based on the FM3D10% 2000 epochs SegVol checkpoint:** Download our preliminary [**checkpoint** (2000 epochs ckpt)](https://drive.google.com/file/d/1dgM5slT5kDV3D_6k_vGpGqU5yB1nTwCL/view?usp=drive_link) first and set the `resume_checkpoint` var in train.py. (Trained on 10% dataset and support box prompt only)

   **Train based on original SegVol checkpoint:** Download the original [**checkpoint**](https://drive.google.com/file/d/1FPj_tiITss5vJF91SrfPEURH6CUEmo4u/view?usp=sharing) first and set the `resume_checkpoint` var in train.py.
3. Download [SegFM](https://www.codabench.org/competitions/5263/) ALL or 10% trainset, set `train_root_path` var in train.py.
4. Download [SegFM](https://www.codabench.org/competitions/5263/) validation set and build **val_samples.json** file to index validation npz files like this:

(Sampling to reduce the val duration)
```
[
"/path/to/xxx.npz",
"/path/to/yyy.npz"
]
```
4. Run
```
torchrun --nproc_per_node=N train.py
```
