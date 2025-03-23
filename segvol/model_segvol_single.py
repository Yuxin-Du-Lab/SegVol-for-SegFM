from transformers import PreTrainedModel, PretrainedConfig
import numpy as np
import monai.transforms as transforms
import nibabel as nib
from scipy import sparse
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
class SegVolConfig(PretrainedConfig):
    model_type = "segvol"

    def __init__(
        self,
        test_mode=True,
        **kwargs,
    ):
        self.spatial_size = [32, 256, 256]
        self.patch_size = [4, 16, 16]
        self.test_mode = test_mode
        super().__init__(**kwargs)

class SegVolModel(PreTrainedModel):
    config_class = SegVolConfig

    def __init__(self, config):
        super().__init__(config)
        sam_model = _build_sam(
            image_encoder_type='vit',
            embed_dim = 768,
            patch_size=self.config.patch_size,
            checkpoint=None,
            image_size=self.config.spatial_size,
        )
        self.model = SegVol(
            image_encoder=sam_model.image_encoder, 
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
            roi_size=self.config.spatial_size,
            patch_size=self.config.patch_size,
            # clip_model=self.config.clip_model,
            test_mode=self.config.test_mode,
            )
        
        self.processor = SegVolProcessor(spatial_size=self.config.spatial_size)
        
        # 添加find_unused_parameters参数
        self.find_unused_parameters = True

    def forward_test(self, 
                image, 
                zoomed_image=None, 
                text_prompt=None, 
                bbox_prompt_group=None, 
                point_prompt_group=None, 
                use_zoom=True,):
        device = image.device
        assert image.shape[0] == 1 and zoomed_image.shape[0] == 1, 'batch size should be 1'
        assert not (text_prompt is None and bbox_prompt_group is None and point_prompt_group is None), 'Drive SegVol using at least one type of prompt'
        bbox_prompt, bbox_prompt_map, point_prompt, point_prompt_map=None, None, None, None
        if bbox_prompt_group is not None:
            bbox_prompt, bbox_prompt_map = bbox_prompt_group
        if point_prompt_group is not None:
            point_prompt, point_prompt_map = point_prompt_group
        volume_shape = image[0][0].shape

        with torch.no_grad():
            logits_global_single = self.model(zoomed_image,
                                                text=text_prompt, 
                                                boxes=bbox_prompt, 
                                                points=point_prompt)
        logits_global_single = F.interpolate(
            logits_global_single.cpu(),
            size=volume_shape, mode='nearest')
        if not use_zoom:
            return logits_global_single
        
        if point_prompt_map is not None:
            binary_points = F.interpolate(
                point_prompt_map.float(),
                size=volume_shape, mode='nearest')
        assert bbox_prompt_map.shape[2:] == volume_shape, 'bbox_prompt_map shape should be the same as volume_shape'
        binary_cube = bbox_prompt_map
        # if bbox_prompt_map is not None:
        #     binary_cube = F.interpolate(
        #         bbox_prompt_map.float(),
        #         size=volume_shape, mode='nearest')
        
        min_d, min_h, min_w, max_d, max_h, max_w = logits2roi_coor(self.config.spatial_size, logits_global_single[0][0], bbox_prompt_map[0][0])
        if min_d is None:
            print('Fail to detect foreground!')
            return logits_global_single

        # Crop roi
        image_single_cropped = image[:, :, min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
        global_preds = (torch.sigmoid(logits_global_single[:, :, min_d:max_d+1, min_h:max_h+1, min_w:max_w+1])>0.5).long()
        
        # assert not (bbox_prompt is not None and point_prompt is not None), 'Do not use point prompt and box prompt at the same time.'
        prompt_reflection = {
            'bbox_prompt_map': None,
            'point_prompt_map': None,
            'global_preds': global_preds
        }
        if bbox_prompt is not None:
            binary_cube_cropped = binary_cube[:, :, min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
            prompt_reflection['bbox_prompt_map'] = binary_cube_cropped
        if point_prompt is not None:
            binary_points_cropped = binary_points[:, :, min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
            prompt_reflection['point_prompt_map'] = binary_points_cropped
        
        ## inference
        with torch.no_grad():
            logits_single_cropped = sliding_window_inference(
                    image_single_cropped.to(device), prompt_reflection,
                    self.config.spatial_size, 1, self.model, 0.5,
                    text=text_prompt,
                    use_box=bbox_prompt is not None,
                    use_point=point_prompt is not None,
                )
            logits_single_cropped = logits_single_cropped.cpu().squeeze()
        logits_global_single[:, :, min_d:max_d+1, min_h:max_h+1, min_w:max_w+1] = logits_single_cropped
        return logits_global_single

    def forward_train(self, image, train_organs, train_labels):
        loss = self.model(image, text=None, boxes=None, points=None,
                 train_organs=train_organs,
                 train_labels=train_labels)
        return loss

    def forward(self, **kwargs):
        if self.config.test_mode:
            return self.forward_test(kwargs['image'], 
                                    kwargs['zoomed_image'], 
                                    kwargs['text_prompt'], 
                                    kwargs['bbox_prompt_group'], 
                                    kwargs['point_prompt_group'], 
                                    kwargs['use_zoom'])
        else:
            return self.forward_train(kwargs['image'],
                                      kwargs['train_organs'],
                                      kwargs['train_labels'])

# FM3D make box
def mask2D_to_bbox(gt2D, bbox_shift=None):
    """将2D mask转换为bbox坐标,使用tensor实现"""
    y_indices, x_indices = torch.where(gt2D > 0)
    if len(x_indices) == 0:
        return torch.tensor([-1, -1, -1, -1])
        
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    
    H, W = gt2D.shape
    if bbox_shift is None:
        bbox_shift = 0
    else:
        bbox_shift = torch.randint(0, bbox_shift, (1,))[0]
    
    scale_y, scale_x = gt2D.shape
    bbox_shift_x = int(bbox_shift * scale_x/256)
    bbox_shift_y = int(bbox_shift * scale_y/256)
    
    x_min = torch.clamp(x_min - bbox_shift_x, min=0)
    x_max = torch.clamp(x_max + bbox_shift_x, max=W-1) 
    y_min = torch.clamp(y_min - bbox_shift_y, min=0)
    y_max = torch.clamp(y_max + bbox_shift_y, max=H-1)
    
    boxes = torch.tensor([x_min, y_min, x_max, y_max])
    return boxes

def mask3D_to_bbox(gt3D, bbox_shift=None):
    """将3D mask转换为bbox坐标和binary cube,使用tensor实现"""
    b_dict = {}
    z_indices, y_indices, x_indices = torch.where(gt3D > 0)
    if len(z_indices) == 0:
        # print('Fail to detect foreground! mask3D_to_bbox')
        return torch.tensor([-1,-1,-1,-1,-1,-1])
        
    z_min, z_max = z_indices.min(), z_indices.max()
    z_middle = z_indices[len(z_indices)//2]
    D, H, W = gt3D.shape
    
    b_dict['z_min'] = z_min.item()
    b_dict['z_max'] = z_max.item()
    b_dict['z_mid'] = z_middle.item()

    gt_mid = gt3D[z_middle]
    box_2d = mask2D_to_bbox(gt_mid, bbox_shift)
    x_min, y_min, x_max, y_max = box_2d
    
    b_dict['z_mid_x_min'] = x_min.item()
    b_dict['z_mid_y_min'] = y_min.item() 
    b_dict['z_mid_x_max'] = x_max.item()
    b_dict['z_mid_y_max'] = y_max.item()

    assert z_min == torch.clamp(z_min, min=0)
    assert z_max == torch.clamp(z_max, max=D-1)
    return torch.tensor([b_dict['z_min'], b_dict['z_mid_y_min'], b_dict['z_mid_x_min'],
                        b_dict['z_max'], b_dict['z_mid_y_max'], b_dict['z_mid_x_max']])

    # return b_dict, \
    #        torch.tensor([b_dict['z_min'], b_dict['z_mid_y_min'], b_dict['z_mid_x_min'],
    #                     b_dict['z_max'], b_dict['z_mid_y_max'], b_dict['z_mid_x_max']]), \
    #        build_binary_cube_dict(b_dict, gt3D.shape)

# processor
class SegVolProcessor():
    def __init__(self, spatial_size) -> None:
        self.img_loader = transforms.LoadImage()
        self.transform4test = transforms.Compose(
            [
                # DimTranspose(keys=["image", "label"]),
                MinMaxNormalization(),
                transforms.CropForegroundd(keys=["image", "label", "cube_boxes"], source_key="image"),
                transforms.ToTensord(keys=["image", "label", "cube_boxes"]),
            ]
        )
        self.zoom_out_transform = transforms.Resized(keys=["image", "label", "cube_boxes"], spatial_size=spatial_size, mode='nearest')
        self.transform4train = transforms.Compose(
        [
            # transforms.AddChanneld(keys=["image"]),
            # DimTranspose(keys=["image", "label"]),
            MinMaxNormalization(),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=spatial_size, mode='constant'),
            # transforms.Resized(keys=["image", "label"], spatial_size=spatial_size),
            transforms.OneOf(transforms=[
                transforms.Resized(keys=["image", "label"], spatial_size=spatial_size),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=spatial_size,
                    pos=3,
                    neg=1,
                    num_samples=1,
                    image_key="image",
                    image_threshold=0,
                ),
                # transforms.RandSpatialCropd(
                #     keys=["image", "label"], roi_size=spatial_size, random_size=False
                # )
                ],
                weights=[1, 3]
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=0.2),
            transforms.RandShiftIntensityd(keys="image", offsets=0.2, prob=0.2),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    
    # ct_path is path for a ct scan file with nii.gz format
    # gt_path is path for a ground truth file with nii.gz format
    def preprocess_ct(self, imgs):
        ct_voxel_ndarray = imgs
        ct_voxel_ndarray = np.array(ct_voxel_ndarray, dtype=np.float32)
        ct_voxel_ndarray = np.expand_dims(ct_voxel_ndarray, axis=0)
        ct_voxel_ndarray = self.ForegroundNorm(ct_voxel_ndarray)
        return ct_voxel_ndarray
    

    def preprocess_gt(self, gts, selected_lab):
        gt_voxel_ndarray = gts
        gt_voxel_ndarray = np.array(gt_voxel_ndarray, dtype=np.int32)
        gt_voxel_ndarray[gt_voxel_ndarray!=selected_lab] = 0
        gt_voxel_ndarray[gt_voxel_ndarray==selected_lab] = 1
        gt_voxel_ndarray = np.expand_dims(gt_voxel_ndarray, axis=0)
        return gt_voxel_ndarray

    def preprocess_ct_gt(self, imgs, gts):
        item = {}
        # generate ct_voxel_ndarray
        ct_voxel_ndarray = imgs
        ct_voxel_ndarray = np.array(ct_voxel_ndarray, dtype=np.float32)
        ct_shape = ct_voxel_ndarray.shape
        ct_voxel_ndarray = np.expand_dims(ct_voxel_ndarray, axis=0)
        ct_voxel_ndarray = self.ForegroundNorm(ct_voxel_ndarray)
        item['image'] = ct_voxel_ndarray

        # generate gt_voxel_ndarray
        gt_voxel_ndarray = gts
        gt_voxel_ndarray = np.array(gt_voxel_ndarray)
        present_categories = np.unique(gt_voxel_ndarray)
        gt_masks = []
        for selected_cls in present_categories:
            if selected_cls == 0:
                continue
            tmp_gt_voxel_ndarray = gt_voxel_ndarray.copy()
            tmp_gt_voxel_ndarray[tmp_gt_voxel_ndarray != selected_cls] = 0
            tmp_gt_voxel_ndarray[tmp_gt_voxel_ndarray == selected_cls] = 1
            gt_masks.append(tmp_gt_voxel_ndarray)
        gt_voxel_ndarray = np.stack(gt_masks, axis=0)
        assert gt_voxel_ndarray.shape[0] == (len(present_categories)-1) and gt_voxel_ndarray.shape[1:] == ct_voxel_ndarray.shape[1:]
        item['label'] = gt_voxel_ndarray.astype(np.int32)

        # transform
        return item['image'], item['label']
    
    # ct_path is path for a ct scan file with nii.gz format
    # gt_path is path for a ground truth file with nii.gz format
    def preprocess_ct_gt_old(self, ct_path, gt_path, category):
        item = {}
        # generate ct_voxel_ndarray
        ct_voxel_ndarray, _ = self.img_loader(ct_path)
        ct_voxel_ndarray = np.array(ct_voxel_ndarray).squeeze()
        ct_shape = ct_voxel_ndarray.shape
        ct_voxel_ndarray = np.expand_dims(ct_voxel_ndarray, axis=0)
        ct_voxel_ndarray = self.ForegroundNorm(ct_voxel_ndarray)
        item['image'] = ct_voxel_ndarray

        # generate gt_voxel_ndarray
        gt_voxel_ndarray, _ = self.img_loader(gt_path)
        gt_voxel_ndarray = np.array(gt_voxel_ndarray)
        present_categories = np.unique(gt_voxel_ndarray)
        gt_masks = []
        for cls_idx in range(len(category)):
            # ignore background
            cls = cls_idx + 1
            if cls not in present_categories:
                gt_voxel_ndarray_category = np.zeros(ct_shape)
                gt_masks.append(gt_voxel_ndarray_category)
            else:
                gt_voxel_ndarray_category = gt_voxel_ndarray.copy()
                gt_voxel_ndarray_category[gt_voxel_ndarray != cls] = 0
                gt_voxel_ndarray_category[gt_voxel_ndarray == cls] = 1
                gt_masks.append(gt_voxel_ndarray_category)
        gt_voxel_ndarray = np.stack(gt_masks, axis=0)
        assert gt_voxel_ndarray.shape[0] == len(category) and gt_voxel_ndarray.shape[1:] == ct_voxel_ndarray.shape[1:]
        item['label'] = gt_voxel_ndarray.astype(np.int32)

        # transform
        return item['image'], item['label']

    def load_uniseg_case(self, ct_npy_path, gt_npy_path):
        img_array = np.load(ct_npy_path)
        allmatrix_sp= sparse.load_npz(gt_npy_path)
        if 'mask_' in gt_npy_path:
            gt_shape = ast.literal_eval(gt_npy_path.split('_')[-1].replace('.npz', ''))
        else:
            gt_shape = ast.literal_eval(gt_npy_path.split('.')[-2])
        gt_array=allmatrix_sp.toarray().reshape(gt_shape)
        return img_array, gt_array

    def ForegroundNorm(self, ct_narray):
        ct_voxel_ndarray = ct_narray.copy()
        ct_voxel_ndarray = ct_voxel_ndarray.flatten()
        thred = np.mean(ct_voxel_ndarray)
        voxel_filtered = ct_voxel_ndarray[(ct_voxel_ndarray > thred)]
        upper_bound = np.percentile(voxel_filtered, 99.95)
        lower_bound = np.percentile(voxel_filtered, 00.05)
        mean = np.mean(voxel_filtered)
        std = np.std(voxel_filtered)
        ct_narray = np.clip(ct_narray, lower_bound, upper_bound)
        ct_narray = (ct_narray - mean) / max(std, 1e-8)
        return ct_narray

    def zoom_transform(self, ct_npy, gt_npy, cube_boxes):
        item = {
            'image': ct_npy,
            'label': gt_npy,
            'cube_boxes': cube_boxes
        }
        item = self.transform4test(item)
        item_zoom_out = self.zoom_out_transform(item)
        item['zoom_out_image'] = item_zoom_out['image']
        item['zoom_out_label'] = item_zoom_out['label']
        item['zoom_out_cube_boxes'] = item_zoom_out['cube_boxes']
        return item

    def point_prompt_b(self, label_single_resize, num_positive_extra=4, num_negative_extra=4, device='cpu'):
        point, point_label = select_points(label_single_resize, num_positive_extra=num_positive_extra, num_negative_extra=num_negative_extra)
        points_single = (point.unsqueeze(0).float().to(device), point_label.unsqueeze(0).float().to(device)) 
        binary_points_resize = build_binary_points(point, point_label, label_single_resize.shape).unsqueeze(0).unsqueeze(0)
        return points_single, binary_points_resize
    
    def bbox_prompt_b(self, label_single_resize, device='cpu'):
        box_single = generate_box(label_single_resize).unsqueeze(0).float().to(device)
        # binary_cube_resize = build_binary_cube(box_single, binary_cube_shape=label_single_resize.shape).unsqueeze(0).unsqueeze(0)
        return box_single

    def dice_score(self, preds, labels, device='cpu'):
        assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match\n" + str(preds.shape) + str(labels.shape)
        predict = preds.view(1, -1).to(device)
        target = labels.view(1, -1).to(device)

        predict = torch.sigmoid(predict)
        predict = torch.where(predict > 0.5, 1., 0.)
        
        tp = torch.sum(torch.mul(predict, target))
        den = torch.sum(predict) + torch.sum(target) + 1
        dice = 2 * tp / den
        return dice
    
    def save_preds(self, ct_path, save_path, logits_mask, start_coord, end_coord):
        ct = nib.load(ct_path)
        logits_mask = logits_mask.transpose(-1, -3)
        start_coord[-1], start_coord[-3] = start_coord[-3], start_coord[-1]
        end_coord[-1], end_coord[-3] = end_coord[-3], end_coord[-1]
        preds_save = torch.zeros(ct.shape)
        preds_save[start_coord[0]:end_coord[0], 
                        start_coord[1]:end_coord[1], 
                        start_coord[2]:end_coord[2]] = torch.sigmoid(logits_mask)
        preds_save = torch.where(preds_save > 0.5, 1., 0.).numpy()
        preds_nii = nib.Nifti1Image(preds_save, affine=ct.affine, header=ct.header)
        nib.save(preds_nii, save_path)

    def train_transform(self, ct_npy, gt_npy):
        item = {
            'image': ct_npy,
            'label': gt_npy
        }
        item = self.transform4train(item)
        if type(item) is list:
            assert len(item) == 1
            item = item[0]
        return item

class MinMaxNormalization(transforms.Transform):
    def __call__(self, data):
        d = dict(data)
        k = "image"
        d[k] = d[k] - d[k].min()
        d[k] = d[k] / np.clip(d[k].max(), a_min=1e-8, a_max=None)
        return d

class DimTranspose(transforms.Transform):
    def __init__(self, keys):
        self.keys = keys
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.swapaxes(d[key], -1, -3)
        return d

# prompts
def generate_box(pred_pre, bbox_shift=None):
    return mask3D_to_bbox(pred_pre, bbox_shift)
    # meaning_post_label = pred_pre # [h, w, d]
    # ones_idx = (meaning_post_label > 0).nonzero(as_tuple=True)
    # if all(tensor.nelement() == 0 for tensor in ones_idx):
    #     bboxes = torch.tensor([-1,-1,-1,-1,-1,-1])
    #     return bboxes
    # min_coords = [dim.min() for dim in ones_idx]    # [x_min, y_min, z_min]
    # max_coords = [dim.max() for dim in ones_idx]    # [x_max, y_max, z_max]


    # if bbox_shift is None:
    #     corner_min = []
    #     corner_max = []
    #     shape = meaning_post_label.shape
    #     for coor in min_coords:
    #         coor_ = max(0, coor)
    #         corner_min.append(coor_)
    #     for idx, coor in enumerate(max_coords):
    #         coor_ = min(shape[idx], coor)
    #         corner_max.append(coor_)
    #     corner_min = torch.tensor(corner_min)
    #     corner_max = torch.tensor(corner_max)
    #     return torch.cat((corner_min, corner_max), dim=0)
    # else:
    #     # add perturbation to bounding box coordinates
    #     corner_min = []
    #     corner_max = []
    #     shape = meaning_post_label.shape
    #     for coor in min_coords:
    #         coor_ = max(0, coor + random.randint(-bbox_shift, bbox_shift))
    #         corner_min.append(coor_)
    #     for idx, coor in enumerate(max_coords):
    #         coor_ = min(shape[idx], coor + random.randint(-bbox_shift, bbox_shift))
    #         corner_max.append(coor_)
    #     corner_min = torch.tensor(corner_min)
    #     corner_max = torch.tensor(corner_max)
    #     return torch.cat((corner_min, corner_max), dim=0)

def build_overlap_mask(logits, train_labels): # logits: (B, 1, H, W, D), train_labels: (B, H, W, D)
    logits = torch.sigmoid(logits[:, 0, :, :, :])
    preds = (logits > 0.5).float()
    labels = train_labels.float()

    # 创建重叠关系掩码overlap_mask均为-1
    overlap_labels = torch.ones_like(preds) * -1
    # preds为1且labels为0的区域设为0
    overlap_labels[(preds == 1) & (labels == 0)] = 0
    # preds为0且labels为1的区域设为1 
    overlap_labels[(preds == 0) & (labels == 1)] = 1
    return overlap_labels

def build_refine_point_prompt(logits, train_labels): # logits: (B, 1, H, W, D), train_labels: (B, H, W, D)
    device = logits.device
    bs = logits.shape[0]

    overlap_labels = build_overlap_mask(logits, train_labels)

    # generate point prompts
    iter_points_ax = []
    iter_point_labels = []
    
    for sample_idx in range(bs):
        # point prompt
        num_positive_extra_max, num_negative_extra_max = 5, 5
        num_positive_extra = random.randint(0, num_positive_extra_max)
        num_negative_extra = random.randint(0, num_negative_extra_max)
        point, point_label = select_points(
            overlap_labels[sample_idx],
            num_positive_extra=num_positive_extra,
            num_negative_extra=num_negative_extra,
            fix_extra_point_num=num_positive_extra_max + num_negative_extra_max)
        iter_points_ax.append(point)
        iter_point_labels.append(point_label)
    # batched prompt
    iter_points_ax = torch.stack(iter_points_ax, dim=0).to(device)
    iter_point_labels = torch.stack(iter_point_labels, dim=0).to(device)
    iter_points = (iter_points_ax, iter_point_labels)
    return iter_points

def select_points(preds, num_positive_extra=4, num_negative_extra=0, fix_extra_point_num=None):
    spacial_dim = 3
    points = torch.zeros((0, 3))
    labels = torch.zeros((0))
    pos_thred = 0.9
    neg_thred = 0.1
    
    # get pos/net indices
    positive_indices = torch.nonzero((preds > pos_thred) & (preds <= 1), as_tuple=True) # ([pos x], [pos y], [pos z])
    negative_indices = torch.nonzero((preds < neg_thred) & (preds >= 0), as_tuple=True)

    ones_idx = (preds > pos_thred).nonzero(as_tuple=True)
    if all(tmp.nelement() == 0 for tmp in ones_idx):
        # all neg
        num_positive_extra = 0
        selected_positive_point = torch.tensor([-1,-1,-1]).unsqueeze(dim=0)
        points = torch.cat((points, selected_positive_point), dim=0)
        labels = torch.cat((labels, torch.tensor([-1]).reshape(1)))
    else:
        # random select a pos point
        random_idx = torch.randint(len(positive_indices[0]), (1,))
        selected_positive_point = torch.tensor([positive_indices[i][random_idx] for i in range(spacial_dim)]).unsqueeze(dim=0)
        points = torch.cat((points, selected_positive_point), dim=0)
        labels = torch.cat((labels, torch.ones((1))))

    if num_positive_extra > 0:
        pos_idx_list = torch.randperm(len(positive_indices[0]))[:num_positive_extra]
        extra_positive_points = []
        for pos_idx in pos_idx_list:
            extra_positive_points.append([positive_indices[i][pos_idx] for i in range(spacial_dim)])
        extra_positive_points = torch.tensor(extra_positive_points).reshape(-1, 3)
        points = torch.cat((points, extra_positive_points), dim=0)
        labels = torch.cat((labels, torch.ones((extra_positive_points.shape[0]))))

    if num_negative_extra > 0:
        neg_idx_list = torch.randperm(len(negative_indices[0]))[:num_negative_extra]
        extra_negative_points = []
        for neg_idx in neg_idx_list:
            extra_negative_points.append([negative_indices[i][neg_idx] for i in range(spacial_dim)])
        extra_negative_points = torch.tensor(extra_negative_points).reshape(-1, 3)
        points = torch.cat((points, extra_negative_points), dim=0)
        labels = torch.cat((labels, torch.zeros((extra_negative_points.shape[0]))))
    
    if fix_extra_point_num is None:
        left_point_num = num_positive_extra + num_negative_extra + 1 - labels.shape[0]
    else:
        left_point_num = fix_extra_point_num  + 1 - labels.shape[0]

    for _ in range(left_point_num):
        ignore_point = torch.tensor([-1,-1,-1]).unsqueeze(dim=0)
        points = torch.cat((points, ignore_point), dim=0)
        labels = torch.cat((labels, torch.tensor([-1]).reshape(1)))

    return points, labels
# SegVol
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import CLIPTextModel, CLIPTextConfig
import random

#%% set up model
class SegVol(nn.Module):
    def __init__(self, 
                image_encoder, 
                mask_decoder,
                prompt_encoder,
                roi_size,
                patch_size,
                test_mode=False,
                ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.text_encoder = TextEncoder()
        self.feat_shape = np.array(roi_size)/np.array(patch_size)
        self.test_mode = test_mode
        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()
        self.decoder_iter = 6

    def forward(self, image, text=None, boxes=None, points=None, **kwargs):
        bs = image.shape[0]
        img_shape = (image.shape[2], image.shape[3], image.shape[4])
        image_embedding, _ = self.image_encoder(image)
        image_embedding = image_embedding.transpose(1, 2).view(bs, -1, 
            int(self.feat_shape[0]), int(self.feat_shape[1]), int(self.feat_shape[2]))
        # test mode
        if self.test_mode:
            return self.forward_decoder(image_embedding, img_shape, text, boxes, points)
        
        # train mode
        ## sl
        sl_loss = self.supervised_forward(image, image_embedding, img_shape, kwargs['train_organs'], kwargs['train_labels'])
        return sl_loss

    def forward_decoder(self, image_embedding, img_shape, text=None, boxes=None, points=None):
        device = image_embedding.device
        with torch.no_grad():
            if boxes is not None:
                if len(boxes.shape) == 2:
                    boxes = boxes[:, None, :] # (B, 1, 6)
            if text is not None:
                text_embedding = self.text_encoder(text, device)  # (B, 768)
            else:
                text_embedding = None
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None,
            text_embedding=text_embedding,
        )

        dense_pe = self.prompt_encoder.get_dense_pe()
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,
            text_embedding = text_embedding,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
          )
        logits = F.interpolate(low_res_masks, size=img_shape, mode='trilinear', align_corners=False)
        return logits

    def supervised_forward(self, image, image_embedding, img_shape, training_organs, train_labels):
        device = image_embedding.device
        iter_points, iter_bboxes, iter_organs = self.build_prompt_label(image.shape[0], training_organs, train_labels, device)

        sl_loss = 0
        # term 1
        prompt_1 = [iter_bboxes, None, None]
        bboxes, points, organs = prompt_1
        logits = self.forward_decoder(image_embedding, img_shape, text=organs, boxes=bboxes, points=points)
        iter_points = build_refine_point_prompt(logits, train_labels)
        # cal loss
        sl_loss_dice = self.dice_loss.forward(logits.squeeze().float(), train_labels.squeeze().float())
        sl_loss_bce = self.bce_loss.forward(logits.squeeze().float(), train_labels.squeeze().float())
        sl_loss += sl_loss_dice + sl_loss_bce
        return sl_loss
    
        # term 2
        prompt_2 = [iter_bboxes, iter_points, None]
        bboxes, points, organs = prompt_2
        logits = self.forward_decoder(image_embedding, img_shape, text=organs, boxes=bboxes, points=points)
        # cal loss
        sl_loss_dice = self.dice_loss.forward(logits.squeeze().float(), train_labels.squeeze().float())
        sl_loss_bce = self.bce_loss.forward(logits.squeeze().float(), train_labels.squeeze().float())
        sl_loss += sl_loss_dice + sl_loss_bce
        return sl_loss / 2.0

    def build_prompt_label(self, bs, training_organs, train_labels, device):
        # generate prompt & label
        iter_organs = []
        iter_bboxes = []
        iter_points_ax = []
        iter_point_labels = []
        for sample_idx in range(bs):
            # organ prompt
            iter_organs.append(training_organs)
            # box prompt
            box = generate_box(train_labels[sample_idx], bbox_shift=3)
            iter_bboxes.append(box)
            # # point prompt
            # num_positive_extra_max, num_negative_extra_max = 10, 10
            # num_positive_extra = random.randint(0, num_positive_extra_max)
            # num_negative_extra = random.randint(0, num_negative_extra_max)
            # point, point_label = select_points(
            #     train_labels[sample_idx],
            #     num_positive_extra=num_positive_extra,
            #     num_negative_extra=num_negative_extra,
            #     fix_extra_point_num=num_positive_extra_max + num_negative_extra_max)
            # iter_points_ax.append(point)
            # iter_point_labels.append(point_label)
        # batched prompt
        # iter_points_ax = torch.stack(iter_points_ax, dim=0).to(device)
        # iter_point_labels = torch.stack(iter_point_labels, dim=0).to(device)
        # iter_points = (iter_points_ax, iter_point_labels)
        iter_bboxes = torch.stack(iter_bboxes, dim=0).float().to(device)
        return None, iter_bboxes, iter_organs
    
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        config = CLIPTextConfig()
        self.clip_text_model = CLIPTextModel(config)
        self.tokenizer = None
        self.dim_align = nn.Linear(512, 768)
        # freeze text encoder
        for param in self.clip_text_model.parameters():
            param.requires_grad = False

    def organ2tokens(self, organ_names, device):
        text_list = ['A computerized tomography of a {}.'.format(organ_name) for organ_name in organ_names]
        tokens = self.tokenizer(text_list, padding=True, return_tensors="pt")
        for key in tokens.keys():
            tokens[key] = tokens[key].to(device)
        return tokens
    
    def forward(self, text, device):
        if text is None:
            return None
        if type(text) is str:
            # text is supposed to be list
            text = [text]
        tokens = self.organ2tokens(text, device)
        clip_outputs = self.clip_text_model(**tokens)
        text_embedding = clip_outputs.pooler_output
        text_embedding = self.dim_align(text_embedding)
        return text_embedding

# loss
import torch
import torch.nn as nn

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        target_ = target.clone()
        target_[target == -1] = 0
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match\n" + str(predict.shape) + '\n' + str(target.shape[0])
        predict = predict.contiguous().view(predict.shape[0], -1)
        target_ = target_.contiguous().view(target_.shape[0], -1)

        num = torch.sum(torch.mul(predict, target_), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target_, dim=1) + self.smooth

        dice_score = 2*num / den
        dice_loss = 1 - dice_score

        # dice_loss_avg = dice_loss[target[:,0]!=-1].sum() / dice_loss[target[:,0]!=-1].shape[0]
        dice_loss_avg = dice_loss.sum() / dice_loss.shape[0]

        return dice_loss_avg

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match\n' + str(predict.shape) + '\n' + str(target.shape)
        target_ = target.clone()
        target_[target == -1] = 0

        ce_loss = self.criterion(predict, target_)

        return ce_loss

# monai inference

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
import random

from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.transforms import Resize
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    ensure_tuple,
    fall_back_tuple,
    look_up_option,
    optional_import,
)

tqdm, _ = optional_import("tqdm", name="tqdm")

__all__ = ["sliding_window_inference"]

def logits2roi_coor(spatial_size, logits_global_single, box_prompt_map):
    # crop predict
    pred_global_single = torch.sigmoid(logits_global_single) > 0.5
    pred_global_single = pred_global_single + box_prompt_map
    ## get all pos idx
    nonzero_indices = torch.nonzero(pred_global_single)
    if nonzero_indices.shape[0] == 0:
        return None, None, None, None, None, None
    ## get boundary
    min_d, max_d = nonzero_indices[:, 0].min(), nonzero_indices[:, 0].max()
    min_h, max_h = nonzero_indices[:, 1].min(), nonzero_indices[:, 1].max()
    min_w, max_w = nonzero_indices[:, 2].min(), nonzero_indices[:, 2].max()
    ## padding
    crop_d, crop_h, crop_w = max_d - min_d + 1, max_h - min_h + 1, max_w - min_w + 1,
    window_d, window_h, window_w = spatial_size
    padding_d, padding_h, padding_w = max(0, window_d-crop_d), max(0, window_h-crop_h), max(0, window_w-crop_w)
    global_d, global_h, global_w = logits_global_single.shape
    min_d = max(0, min_d - int(padding_d)//2)
    min_h = max(0, min_h - int(padding_h)//2)
    min_w = max(0, min_w - int(padding_w)//2)
    max_d = min(global_d, max_d + int(padding_d)//2)
    max_h = min(global_h, max_h + int(padding_h)//2)
    max_w = min(global_w, max_w + int(padding_w)//2)
    return min_d, min_h, min_w, max_d, max_h, max_w

def build_binary_cube(bbox, binary_cube_shape):
    min_coord = bbox[0][:3].int().tolist()
    max_coord = bbox[0][3:].int().tolist()
    binary_cube = torch.zeros(binary_cube_shape)
    binary_cube[min_coord[0]:max_coord[0]+1, min_coord[1]:max_coord[1]+1, min_coord[2]:max_coord[2]+1] = 1
    return binary_cube

def build_binary_cube_dict(bbox, binary_cube_shape):
    min_coord = [bbox['z_min'], bbox['z_mid_y_min'], bbox['z_mid_x_min']]
    max_coord = [bbox['z_max'], bbox['z_mid_y_max'], bbox['z_mid_x_max']]
    binary_cube = torch.zeros(binary_cube_shape)
    binary_cube[min_coord[0]:max_coord[0]+1, 
                min_coord[1]:max_coord[1]+1,
                min_coord[2]:max_coord[2]+1] = 1
    return binary_cube

def build_binary_points(points, labels, shape):
    binary_points = torch.zeros(shape, dtype=torch.int16)
    binary_points[points[labels == 1, 0].long(), points[labels == 1, 1].long(), points[labels == 1, 2].long()] = 1
    return binary_points

def sliding_window_inference(
    inputs: torch.Tensor,
    prompt_reflection: Dict[Any, torch.Tensor],
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor], Dict[Any, torch.Tensor]]],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    progress: bool = False,
    roi_weight_map: Union[torch.Tensor, None] = None,
    *args: Any,
    **kwargs: Any,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[Any, torch.Tensor]]:
    """
    Sliding window inference on `inputs` with `predictor`.

    The outputs of `predictor` could be a tensor, a tuple, or a dictionary of tensors.
    Each output in the tuple or dict value is allowed to have different resolutions with respect to the input.
    e.g., the input patch spatial size is [128,128,128], the output (a tuple of two patches) patch sizes
    could be ([128,64,256], [64,32,128]).
    In this case, the parameter `overlap` and `roi_size` need to be carefully chosen to ensure the output ROI is still
    an integer. If the predictor's input and output spatial sizes are not equal, we recommend choosing the parameters
    so that `overlap*roi_size*output_size/input_size` is an integer (for each spatial dimension).

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor ``patch_data`` in shape NCHW[D],
            The outputs of the function call ``predictor(patch_data)`` should be a tensor, a tuple, or a dictionary
            with Tensor values. Each output in the tuple or dict value should have the same batch_size, i.e. NM'H'W'[D'];
            where H'W'[D'] represents the output patch's spatial size, M is the number of output channels,
            N is `sw_batch_size`, e.g., the input shape is (7, 1, 128,128,128),
            the output could be a tuple of two tensors, with shapes: ((7, 5, 128, 64, 256), (7, 4, 64, 32, 128)).
            In this case, the parameter `overlap` and `roi_size` need to be carefully chosen
            to ensure the scaled output ROI sizes are still integers.
            If the `predictor`'s input and output spatial sizes are different,
            we recommend choosing the parameters so that ``overlap*roi_size*zoom_scale`` is an integer for each dimension.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        progress: whether to print a `tqdm` progress bar.
        roi_weight_map: pre-computed (non-negative) weight map for each ROI.
            If not given, and ``mode`` is not `constant`, this map will be computed on the fly.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    print('sliding window inference for ROI')
    text = kwargs['text']
    use_box = kwargs['use_box']
    use_point = kwargs['use_point']
    # assert not (use_box and use_point)
    compute_dtype = inputs.dtype
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    batch_size, _, *image_size_ = inputs.shape

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)
    #############
    if use_point or use_box:
        global_preds = prompt_reflection['global_preds']
        binary_prompt_map_box = prompt_reflection['bbox_prompt_map']
        binary_prompt_map_point = prompt_reflection['point_prompt_map']
        global_preds = F.pad(global_preds, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)
        if binary_prompt_map_box is not None:
            binary_prompt_map_box = F.pad(binary_prompt_map_box, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)
        if binary_prompt_map_point is not None:
            binary_prompt_map_point = F.pad(binary_prompt_map_point, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)
    #############
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    valid_patch_size = get_valid_patch_size(image_size, roi_size)
    if valid_patch_size == roi_size and (roi_weight_map is not None):
        importance_map = roi_weight_map
    else:
        try:
            importance_map = compute_importance_map(valid_patch_size, mode=mode, sigma_scale=sigma_scale, device=device)
        except BaseException as e:
            raise RuntimeError(
                "Seems to be OOM. Please try smaller patch size or mode='constant' instead of mode='gaussian'."
            ) from e
    importance_map = convert_data_type(importance_map, torch.Tensor, device, compute_dtype)[0]  # type: ignore
    # handle non-positive weights
    min_non_zero = max(importance_map[importance_map != 0].min().item(), 1e-3)
    importance_map = torch.clamp(importance_map.to(torch.float32), min=min_non_zero).to(compute_dtype)

    # Perform predictions
    dict_key, output_image_list, count_map_list = None, [], []
    _initialized_ss = -1
    is_tensor_output = True  # whether the predictor's output is a tensor (instead of dict/tuple)

    # for each patch
    for slice_g in tqdm(range(0, total_slices, sw_batch_size)) if progress else range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        #############
        
        boxes = None
        points = None
        if use_point:
            window_binary_prompt_map = torch.cat([binary_prompt_map_point[win_slice] for win_slice in unravel_slice]).to(sw_device)
            point, point_label = select_points(window_binary_prompt_map.squeeze())
            points = (point.unsqueeze(0).float().to(device), point_label.unsqueeze(0).float().to(device))  
            pseudo_label = torch.cat([global_preds[win_slice] for win_slice in unravel_slice]).to(sw_device)
            boxes = generate_box(pseudo_label.squeeze()).unsqueeze(0).float().to(device)
        if use_box:
            if num_win == 1:
                window_binary_prompt_map = torch.cat([binary_prompt_map_box[win_slice] for win_slice in unravel_slice]).to(sw_device)
                boxes = generate_box(window_binary_prompt_map.squeeze()).unsqueeze(0).float().to(device)
            else:
                pseudo_label = torch.cat([global_preds[win_slice] for win_slice in unravel_slice]).to(sw_device)
                boxes = generate_box(pseudo_label.squeeze()).unsqueeze(0).float().to(device)
        seg_prob_out = predictor(window_data, text, boxes, points)  # batched patch segmentation
        #############
        # convert seg_prob_out to tuple seg_prob_tuple, this does not allocate new memory.
        seg_prob_tuple: Tuple[torch.Tensor, ...]
        if isinstance(seg_prob_out, torch.Tensor):
            seg_prob_tuple = (seg_prob_out,)
        elif isinstance(seg_prob_out, Mapping):
            if dict_key is None:
                dict_key = sorted(seg_prob_out.keys())  # track predictor's output keys
            seg_prob_tuple = tuple(seg_prob_out[k] for k in dict_key)
            is_tensor_output = False
        else:
            seg_prob_tuple = ensure_tuple(seg_prob_out)
            is_tensor_output = False

        # for each output in multi-output list
        for ss, seg_prob in enumerate(seg_prob_tuple):
            seg_prob = seg_prob.to(device)  # BxCxMxNxP or BxCxMxN

            # compute zoom scale: out_roi_size/in_roi_size
            zoom_scale = []
            for axis, (img_s_i, out_w_i, in_w_i) in enumerate(
                zip(image_size, seg_prob.shape[2:], window_data.shape[2:])
            ):
                _scale = out_w_i / float(in_w_i)
                if not (img_s_i * _scale).is_integer():
                    warnings.warn(
                        f"For spatial axis: {axis}, output[{ss}] will have non-integer shape. Spatial "
                        f"zoom_scale between output[{ss}] and input is {_scale}. Please pad inputs."
                    )
                zoom_scale.append(_scale)

            if _initialized_ss < ss:  # init. the ss-th buffer at the first iteration
                # construct multi-resolution outputs
                output_classes = seg_prob.shape[1]
                output_shape = [batch_size, output_classes] + [
                    int(image_size_d * zoom_scale_d) for image_size_d, zoom_scale_d in zip(image_size, zoom_scale)
                ]
                # allocate memory to store the full output and the count for overlapping parts
                output_image_list.append(torch.zeros(output_shape, dtype=compute_dtype, device=device))
                count_map_list.append(torch.zeros([1, 1] + output_shape[2:], dtype=compute_dtype, device=device))
                _initialized_ss += 1

            # resizing the importance_map
            resizer = Resize(spatial_size=seg_prob.shape[2:], mode="nearest", anti_aliasing=False)

            # store the result in the proper location of the full output. Apply weights from importance map.
            for idx, original_idx in zip(slice_range, unravel_slice):
                # zoom roi
                original_idx_zoom = list(original_idx)  # 4D for 2D image, 5D for 3D image
                for axis in range(2, len(original_idx_zoom)):
                    zoomed_start = original_idx[axis].start * zoom_scale[axis - 2]
                    zoomed_end = original_idx[axis].stop * zoom_scale[axis - 2]
                    if not zoomed_start.is_integer() or (not zoomed_end.is_integer()):
                        warnings.warn(
                            f"For axis-{axis-2} of output[{ss}], the output roi range is not int. "
                            f"Input roi range is ({original_idx[axis].start}, {original_idx[axis].stop}). "
                            f"Spatial zoom_scale between output[{ss}] and input is {zoom_scale[axis - 2]}. "
                            f"Corresponding output roi range is ({zoomed_start}, {zoomed_end}).\n"
                            f"Please change overlap ({overlap}) or roi_size ({roi_size[axis-2]}) for axis-{axis-2}. "
                            "Tips: if overlap*roi_size*zoom_scale is an integer, it usually works."
                        )
                    original_idx_zoom[axis] = slice(int(zoomed_start), int(zoomed_end), None)
                importance_map_zoom = resizer(importance_map.unsqueeze(0))[0].to(compute_dtype)
                # store results and weights
                output_image_list[ss][original_idx_zoom] += importance_map_zoom * seg_prob[idx - slice_g]
                count_map_list[ss][original_idx_zoom] += (
                    importance_map_zoom.unsqueeze(0).unsqueeze(0).expand(count_map_list[ss][original_idx_zoom].shape)
                )

    # account for any overlapping sections
    for ss in range(len(output_image_list)):
        output_image_list[ss] = (output_image_list[ss] / count_map_list.pop(0)).to(compute_dtype)

    # remove padding if image_size smaller than roi_size
    for ss, output_i in enumerate(output_image_list):
        if torch.isnan(output_i).any() or torch.isinf(output_i).any():
            warnings.warn("Sliding window inference results contain NaN or Inf.")

        zoom_scale = [
            seg_prob_map_shape_d / roi_size_d for seg_prob_map_shape_d, roi_size_d in zip(output_i.shape[2:], roi_size)
        ]

        final_slicing: List[slice] = []
        for sp in range(num_spatial_dims):
            slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
            slice_dim = slice(
                int(round(slice_dim.start * zoom_scale[num_spatial_dims - sp - 1])),
                int(round(slice_dim.stop * zoom_scale[num_spatial_dims - sp - 1])),
            )
            final_slicing.insert(0, slice_dim)
        while len(final_slicing) < len(output_i.shape):
            final_slicing.insert(0, slice(None))
        output_image_list[ss] = output_i[final_slicing]

    if dict_key is not None:  # if output of predictor is a dict
        final_output = dict(zip(dict_key, output_image_list))
    else:
        final_output = tuple(output_image_list)  # type: ignore
    return final_output[0] if is_tensor_output else final_output  # type: ignore


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)

# build 3D SAM
import torch
import numpy as np
from monai.networks.nets import ViT

def _build_sam(
    image_encoder_type,
    embed_dim,
    patch_size,
    checkpoint,
    image_size,
):
    mlp_dim = 3072
    num_layers = 12
    num_heads = 12
    pos_embed = 'perceptron'
    dropout_rate = 0.0
    
    image_encoder=ViT(
        in_channels=1,
        img_size=image_size,
        patch_size=patch_size,
        hidden_size=embed_dim,
        mlp_dim=mlp_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        pos_embed=pos_embed,
        classification=False,
        dropout_rate=dropout_rate,
    )
    image_embedding_size = [int(item) for item in (np.array(image_size) / np.array(patch_size))]

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')['state_dict']
            encoder_dict = {k.replace('model.encoder.', ''): v for k, v in state_dict.items() if 'model.encoder.' in k}
        image_encoder.load_state_dict(encoder_dict)
        print(f'===> image_encoder.load_param: {checkpoint}')
    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=image_embedding_size,
            input_image_size=image_size,
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            image_encoder_type=image_encoder_type,
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            image_size=np.array(image_size),
            patch_size=np.array(patch_size),
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    return sam

# mask decoder
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type, Optional

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        image_encoder_type: str,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        image_size,
        patch_size,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        if image_encoder_type == 'swin_vit':
            self.feat_shape = image_size/patch_size
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose3d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                nn.LayerNorm((transformer_dim // 4, int(self.feat_shape[0]), int(self.feat_shape[1]), int(self.feat_shape[2]))),    # swin
                activation(),
                nn.ConvTranspose3d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),        # swin
                # nn.Conv3d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, stride=1, padding=1),    # vit
                activation(),
            )
        else:
            self.feat_shape = image_size/patch_size * 2
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose3d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                nn.LayerNorm((transformer_dim // 4, int(self.feat_shape[0]), int(self.feat_shape[1]), int(self.feat_shape[2]))), # vit
                activation(),
                nn.ConvTranspose3d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                # nn.Conv3d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, stride=1, padding=1),
                activation(),
            )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.txt_align_upscaled_embedding = nn.Linear(768, 96)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embedding: Optional[torch.Tensor],
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Returns:
          torch.Tensor: batched predicted masks
        """
        # print('--------------decoder here--------------')
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            text_embedding=text_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        text_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w, d = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w, d)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w, d = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w * d)).view(b, -1, h, w, d)
        
        if text_embedding is not None:
            text_embedding_down = self.txt_align_upscaled_embedding(text_embedding).unsqueeze(dim=1)
            upscaled_embedding = upscaled_embedding.view(b, c, h * w * d)
            sim = (text_embedding_down @ upscaled_embedding).view(b, -1, h, w, d)
            sim = sim.repeat(1, masks.shape[1], 1, 1, 1)
            masks = masks + sim
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

# prompt encoder
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int, int],
        input_image_size: Tuple[int, int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1], 4 * image_embedding_size[2])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 3), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 3)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        text_embedding: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif text_embedding is not None:
            return text_embedding.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        text_embedding: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        bs = self._get_batch_size(points, boxes, masks, text_embedding)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        
        if text_embedding is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, text_embedding.unsqueeze(dim=1)], dim=1)
        
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1, 1).expand(
                bs, -1, int(self.image_embedding_size[0]), int(self.image_embedding_size[1]), int(self.image_embedding_size[2])
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((3, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w, d = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w, d), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        z_embed = grid.cumsum(dim=2) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        z_embed = z_embed / d

        pe = self._pe_encoding(torch.stack([x_embed, y_embed, z_embed], dim=-1))
        return pe.permute(3, 0, 1, 2)  # C x H x W x D

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        coords[:, :, 2] = coords[:, :, 2] / image_size[2]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

# two way transformer
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w, d = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))
    

# sam
class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        mask_decoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
        """
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        # TODO
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x