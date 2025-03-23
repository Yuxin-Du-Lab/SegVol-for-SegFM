from transformers import AutoTokenizer, AutoConfig
import torch
import os
import numpy as np
from segvol.model_segvol_single import build_binary_cube_dict, SegVolModel
from tqdm import tqdm
import json

device_id = 0
device = torch.device(f"cuda:{device_id}")

class SegDatasetTest(torch.utils.data.Dataset):
    def __init__(self, file_paths, preprocessor):
        self.file_paths = file_paths
        self.preprocessor = preprocessor
        self.epoch = 0
    
    def __len__(self):
        return len(self.file_paths)
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        npz = np.load(file_path, allow_pickle=True)
        imgs = npz['imgs']
        gts = npz['gts']
        
        imgs, gts = self.preprocessor.preprocess_ct_gt(imgs, gts)   # (1, H, W, D), (N, H, W, D)
        boxes = npz['boxes'] # a list of bounding box prompts
        cube_boxes = []
        for std_box in boxes:
            binary_cube = build_binary_cube_dict(std_box, imgs.shape[1:])
            cube_boxes.append(binary_cube)
        cube_boxes = torch.stack(cube_boxes, dim=0)
        assert cube_boxes.shape == gts.shape, f'{cube_boxes.shape} != {gts.shape}'

        zoom_item = self.preprocessor.zoom_transform(imgs, gts, cube_boxes)
        zoom_item['file_path'] = file_path
        return zoom_item



def validation(model_val, val_dataloader, processor):  

    total_bbox_dice = 0.0
    total_samples = 0
    
    for data_item_list in tqdm(val_dataloader, desc="Validating"):
        data_item = data_item_list[0]
        category_num = data_item['label'].shape[0]
        batch_bbox_dice = 0.0
        print(data_item['file_path'])
        data_item['image'], data_item['label'], data_item['zoom_out_image'], data_item['zoom_out_label'] = \
        data_item['image'].unsqueeze(0).to(device), data_item['label'].unsqueeze(0).to(device), data_item['zoom_out_image'].unsqueeze(0).to(device), data_item['zoom_out_label'].unsqueeze(0).to(device)
        
        for cls_idx in range(category_num):
            
            cube_boxes = data_item['cube_boxes'][cls_idx].unsqueeze(0).unsqueeze(0)
            # bbox_prompt = processor.bbox_prompt_b(data_item['zoom_out_label'][0][cls_idx], device=device) 
            bbox_prompt = processor.bbox_prompt_b(data_item['zoom_out_cube_boxes'][cls_idx], device=device) 

            with torch.no_grad():
                logits_mask = model_val.forward_test(
                    image=data_item['image'],
                    zoomed_image=data_item['zoom_out_image'],
                    bbox_prompt_group=[bbox_prompt, cube_boxes],
                    use_zoom=True
                )

                gt_label = data_item['label'][0][cls_idx]
                pred_label = logits_mask[0][0]

                bbox_dice = processor.dice_score(pred_label, gt_label, device)

                print(f'cls {cls_idx} - BBox Dice: {bbox_dice.item():.4f}')
                batch_bbox_dice += bbox_dice.item()

            # clear cache
            torch.cuda.empty_cache()
        total_bbox_dice += batch_bbox_dice
        total_samples += category_num
    avg_bbox_dice = total_bbox_dice / total_samples
    
    print(f'Average BBox Dice: {avg_bbox_dice:.4f}')
    return avg_bbox_dice

if __name__ == '__main__':
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

    with open('val_samples.json', 'r') as f:
        val_file_paths = json.load(f)
    
    val_dataset = SegDatasetTest(val_file_paths, processor)
    print('val dataset size:', len(val_dataset))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x
    )
    validation(model_val, val_dataloader, processor)