---
license: mit
language:
- en
- zh
pipeline_tag: image-segmentation
tags:
- medical
---


![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/6565b54a9bf6665f10f75441/no60wyvKDTD-WV3pCt2P5.jpeg)

Language: [ZH / EN]

SegVol是用于体积医学图像分割的通用交互式模型,可以使用点，框和文本作为prompt驱动模型，输出分割结果。

通过在90k个无标签CT和6k个有标签CT上进行训练，该基础模型支持对200多个解剖类别进行分割。

The SegVol is a universal and interactive model for volumetric medical image segmentation. SegVol accepts point, box, and text prompts while output volumetric segmentation. By training on 90k unlabeled Computed Tomography (CT) volumes and 6k labeled CTs, this foundation model supports the segmentation of over 200 anatomical categories.

[**Paper**](https://arxiv.org/abs/2311.13385), [**Code**](https://github.com/BAAI-DCAI/SegVol) 和 [**Demo**](https://www.modelscope.cn/studios/YuxinDu/SegVol/summary) 已发布。

**Keywords**: 3D medical SAM, volumetric image segmentation

## Quicktart

### Requirements
```
conda create -n segvol_transformers python=3.8
conda activate segvol_transformers
```
需要 [pytorch v1.11.0](https://pytorch.org/get-started/previous-versions/) 或更高版本。另外请安装如下支持包：

[pytorch v1.11.0](https://pytorch.org/get-started/previous-versions/) or higher version is required. Please also install the following support packages:

```
pip install modelscope
pip install 'monai[all]==0.9.0'
pip install einops==0.6.1
pip install transformers==4.18.0
pip install matplotlib
```

### Test script

```python
from transformers import AutoModel, AutoTokenizer
import torch
from modelscope import snapshot_download
import os

model_dir = snapshot_download('yuxindu/SegVol')

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model
clip_tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, test_mode=True)
model.model.text_encoder.tokenizer = clip_tokenizer
model.eval()
model.to(device)
print('model load done')

# set case path
ct_path = os.path.join(model_dir, 'Case_image_00001_0000.nii.gz')
gt_path = os.path.join(model_dir, 'Case_label_00001.nii.gz')

# set categories, corresponding to the unique values(1, 2, 3, 4, ...) in ground truth mask
categories = ["liver", "kidney", "spleen", "pancreas"]

# generate npy data format
ct_npy, gt_npy = model.processor.preprocess_ct_gt(ct_path, gt_path, category=categories)
# IF you have download our 25 processed datasets, you can skip to here with the processed ct_npy, gt_npy files

# go through zoom_transform to generate zoomout & zoomin views
data_item = model.processor.zoom_transform(ct_npy, gt_npy)

# add batch dim manually
data_item['image'], data_item['label'], data_item['zoom_out_image'], data_item['zoom_out_label'] = \
data_item['image'].unsqueeze(0).to(device), data_item['label'].unsqueeze(0).to(device), data_item['zoom_out_image'].unsqueeze(0).to(device), data_item['zoom_out_label'].unsqueeze(0).to(device)

# take liver as the example
cls_idx = 0

# text prompt
text_prompt = [categories[cls_idx]]

# point prompt
point_prompt, point_prompt_map = model.processor.point_prompt_b(data_item['zoom_out_label'][0][cls_idx], device=device)   # inputs w/o batch dim, outputs w batch dim

# bbox prompt
bbox_prompt, bbox_prompt_map = model.processor.bbox_prompt_b(data_item['zoom_out_label'][0][cls_idx], device=device)   # inputs w/o batch dim, outputs w batch dim

print('prompt done')

# segvol test forward
# use_zoom: use zoom-out-zoom-in
# point_prompt_group: use point prompt
# bbox_prompt_group: use bbox prompt
# text_prompt: use text prompt
logits_mask = model.forward_test(image=data_item['image'],
      zoomed_image=data_item['zoom_out_image'],
      # point_prompt_group=[point_prompt, point_prompt_map],
      bbox_prompt_group=[bbox_prompt, bbox_prompt_map],
      text_prompt=text_prompt,
      use_zoom=True
      )

# cal dice score
dice = model.processor.dice_score(logits_mask[0][0], data_item['label'][0][cls_idx], device)
print(dice)

# save prediction as nii.gz file
save_path='./Case_preds_00001.nii.gz'
model.processor.save_preds(ct_path, save_path, logits_mask[0][0], 
                           start_coord=data_item['foreground_start_coord'], 
                           end_coord=data_item['foreground_end_coord'])
print('done')
```

### Training script

```python
from transformers import AutoModel, AutoTokenizer
import torch
from modelscope import snapshot_download
import os

model_dir = snapshot_download('yuxindu/SegVol')

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model
clip_tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, test_mode=False)
model.model.text_encoder.tokenizer = clip_tokenizer
model.train()
model.to(device)
print('model load done')

# set case path
ct_path = os.path.join(model_dir, 'Case_image_00001_0000.nii.gz')
gt_path = os.path.join(model_dir, 'Case_label_00001.nii.gz')

# set categories, corresponding to the unique values(1, 2, 3, 4, ...) in ground truth mask
categories = ["liver", "kidney", "spleen", "pancreas"]

# generate npy data format
ct_npy, gt_npy = model.processor.preprocess_ct_gt(ct_path, gt_path, category=categories)
# IF you have download our 25 processed datasets, you can skip to here with the processed ct_npy, gt_npy files

# go through train transform
data_item = model.processor.train_transform(ct_npy, gt_npy)

# training example
# add batch dim manually
image, gt3D = data_item["image"].unsqueeze(0).to(device), data_item["label"].unsqueeze(0).to(device) # add batch dim

loss_step_avg = 0
for cls_idx in range(len(categories)):
    # optimizer.zero_grad()
    organs_cls = categories[cls_idx]
    labels_cls = gt3D[:, cls_idx]
    loss = model.forward_train(image, train_organs=organs_cls, train_labels=labels_cls)
    loss_step_avg += loss.item()
    loss.backward()
    # optimizer.step()

loss_step_avg /= len(categories)
print(f'AVG loss {loss_step_avg}')

# save ckpt
model.save_pretrained('./ckpt')
```

### Start with M3D-Seg dataset

我们已经发布了用于训练SegVol的25个开源数据集(M3D-Seg),并将预处理后的数据上传到了[ModelScope](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-Seg/summary)和[HuggingFace](https://huggingface.co/datasets/GoodBaiBai88/M3D-Seg)。 
您可以使用下面的script方便地载入，并插入到Test script和Training script中。

We have released 25 open source datasets(M3D-Seg) for training SegVol, and these preprocessed data have been uploaded to [ModelScope](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-Seg/summary) and [HuggingFace](https://huggingface.co/datasets/GoodBaiBai88/M3D-Seg). 
You can use the following script to easily load cases and insert them into Test script and Training script.

```bash
git clone https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-Seg.git
```

```python
import json, os
M3D_Seg_path = 'path/to/M3D-Seg'

# select a dataset
dataset_code = '0000'

# load json dict
json_path = os.path.join(M3D_Seg_path, dataset_code, dataset_code + '.json')
with open(json_path, 'r') as f:
    dataset_dict = json.load(f)

# get a case
ct_path = os.path.join(M3D_Seg_path, dataset_dict['train'][0]['image'])
gt_path = os.path.join(M3D_Seg_path, dataset_dict['train'][0]['label'])

# get categories
categories_dict = dataset_dict['labels']
categories = [x for _, x in categories_dict.items() if x != "background"]

# load npy data format
ct_npy, gt_npy = model.processor.load_uniseg_case(ct_path, gt_path)
```