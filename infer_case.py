from transformers import AutoTokenizer, AutoConfig
import torch
import os
import numpy as np
from segvol.model_segvol_single import build_binary_cube_dict, SegVolModel
from tqdm import tqdm
import json
from glob import glob

def load_item(npz_file, preprocessor):
    file_path = npz_file
    npz = np.load(file_path, allow_pickle=True)
    imgs = npz['imgs']
    gts = npz['gts']
    
    imgs, gts = preprocessor.preprocess_ct_gt(imgs, gts)   # (1, H, W, D), (N, H, W, D)
    boxes = npz['boxes'] # a list of bounding box prompts
    cube_boxes = []
    for std_box in boxes:
        binary_cube = build_binary_cube_dict(std_box, imgs.shape[1:])
        cube_boxes.append(binary_cube)
    cube_boxes = torch.stack(cube_boxes, dim=0)
    assert cube_boxes.shape == gts.shape, f'{cube_boxes.shape} != {gts.shape}'

    zoom_item = preprocessor.zoom_transform(imgs, gts, cube_boxes)
    zoom_item['file_path'] = file_path
    zoom_item['gts_original'] = torch.from_numpy(npz['gts'])
    return zoom_item

def backfill_foreground_preds(ct_shape, logits_mask, start_coord, end_coord):
    binary_preds = torch.zeros(ct_shape)
    binary_preds[start_coord[0]:end_coord[0], 
                    start_coord[1]:end_coord[1], 
                    start_coord[2]:end_coord[2]] = torch.sigmoid(logits_mask)
    binary_preds = torch.where(binary_preds > 0.5, 1., 0.)
    return binary_preds

def infer_case(model_val, data_item, processor, device):
    data_item['image'], data_item['label'], data_item['zoom_out_image'], data_item['zoom_out_label'] = \
    data_item['image'].unsqueeze(0).to(device), data_item['label'].unsqueeze(0).to(device), data_item['zoom_out_image'].unsqueeze(0).to(device), data_item['zoom_out_label'].unsqueeze(0).to(device)
    start_coord, end_coord = data_item['foreground_start_coord'], data_item['foreground_end_coord']

    gts = data_item['gts_original']
    category_ids = torch.unique(gts)
    final_preds = torch.zeros_like(gts)

    for category_id in category_ids:
        if category_id == 0:
            continue
        cls_idx = (category_id - 1).item()
        cube_boxes = data_item['cube_boxes'][cls_idx].unsqueeze(0).unsqueeze(0)
        bbox_prompt = processor.bbox_prompt_b(data_item['zoom_out_cube_boxes'][cls_idx], device=device) 
        with torch.no_grad():
            logits_mask = model_val.forward_test(
                image=data_item['image'],
                zoomed_image=data_item['zoom_out_image'],
                bbox_prompt_group=[bbox_prompt, cube_boxes],
                use_zoom=True
            )
        binary_preds = backfill_foreground_preds(gts.shape, logits_mask, start_coord, end_coord)
        final_preds[binary_preds == 1] = category_id
    
        # clear cache
        torch.cuda.empty_cache()
    return final_preds.numpy()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = './segvol'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    ckpt_path = './epoch_2000_loss_0.2232.pth'
    clip_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model_val = SegVolModel(config)
    model_val.model.text_encoder.tokenizer = clip_tokenizer
    processor = model_val.processor
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_val.load_state_dict(checkpoint['model_state_dict'])
    model_val.eval()
    model_val.to(device)

    npz_file = glob("inputs/*.npz")[0]
    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)

    data_item = load_item(npz_file, processor)
    # dict_keys(['image', 'label', 'cube_boxes', 'foreground_start_coord', 'foreground_end_coord', 
    # 'image_transforms', 'label_transforms', 'cube_boxes_transforms', 'zoom_out_image', 'zoom_out_label', 
    # 'zoom_out_cube_boxes', 'file_path'])
    final_preds = infer_case(model_val, data_item, processor, device)
    output_path = os.path.join(out_dir, os.path.basename(npz_file))
    np.savez_compressed(output_path, segs=final_preds)
    print('done')