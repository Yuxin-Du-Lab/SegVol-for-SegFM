from transformers import AutoTokenizer, AutoConfig
import torch
import os
import glob
import numpy as np
from segvol.model_segvol_single import build_overlap_mask, build_binary_cube_dict, SegVolModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import json
from validation import validation, SegDatasetTest

# torchrun --nproc_per_node=1 train.py

# Add code to restore checkpoint before training loop
def load_checkpoint(checkpoint_path, device, model, optimizer, scheduler, num_epochs, eta_min):
    if not os.path.exists(checkpoint_path):
        if local_rank == 0:
            print(f"Checkpoint not found: {checkpoint_path}")
        return 0, float('inf')
    print(f'load checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint.keys():
        model.module.load_state_dict(checkpoint['model_state_dict'])
        if local_rank == 0 and 'model_state_dict' in checkpoint.keys():
            print(f"Resuming training from epoch {checkpoint['epoch']}")
        
        return checkpoint['epoch'] + 1, checkpoint['loss']
    else:
        # for SegVol_v1.pth
        model_dict = {}
        for k, v in list(checkpoint['model'].items()):
            new_k = k.replace('module.', 'model.')
            model_dict[new_k] = v
        model.module.load_state_dict(model_dict)
        print(f"Resuming training from SegVol_v1.pth")
        return 0, float('inf')
    
    

def setup_ddp():
    # Initialize process group
    dist.init_process_group(backend="nccl")
    # Get local rank and world_size
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# build dataset
class SegDatasetTrain(torch.utils.data.Dataset):
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
        # print(file_path)
        npz = np.load(file_path, allow_pickle=True)
        imgs = npz['imgs']
        gts = npz['gts']
        # train_transform
        imgs = self.preprocessor.preprocess_ct(imgs)
        unique_labs = np.unique(gts)[1:]
        # selected_lab = np.random.choice(unique_labs)
        selected_lab = unique_labs[self.epoch % len(unique_labs)]
        gts = self.preprocessor.preprocess_gt(gts, selected_lab)
        item = self.preprocessor.train_transform(imgs, gts)
        return item

def collate_fn(batch):
    images = []
    label = []
    foreground_start_coord = []
    foreground_end_coord = []

    for sample in batch:
        images.append(sample['image'])
        label.append(sample['label'])
        foreground_start_coord.append(sample['foreground_start_coord'])
        foreground_end_coord.append(sample['foreground_end_coord'])
    return {
        'image': torch.stack(images, dim=0),
        'label': torch.stack(label, dim=0),
        'foreground_start_coord': foreground_start_coord,
        'foreground_end_coord': foreground_end_coord
    }

def val_model(local_rank, model_dir, ckpt_path):
    if local_rank != 0:  # Only validate on the main process
        return 0.0, 0.0  # Return two 0.0 values for the two dice scores
        
    print(f'validation on {ckpt_path}')
    clip_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, test_mode=True)
    model_val = SegVolModel(config)
    model_val.model.text_encoder.tokenizer = clip_tokenizer
    processor = model_val.processor

    device = torch.device(f"cuda:{local_rank}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in checkpoint.keys():
        model_val.load_state_dict(checkpoint['model_state_dict'])
    else:
        # for SegVol_v1.pth
        model_dict = {}
        for k, v in list(checkpoint['model'].items()):
            new_k = k.replace('module.', 'model.')
            model_dict[new_k] = v
        model_val.load_state_dict(model_dict)
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

    avg_bbox_dice = validation(model_val, val_dataloader, processor)
    return avg_bbox_dice
    

def train_one_epoch(model, train_dataloader, optimizer, epoch):
    model.train()
    total_loss = 0
    with tqdm(train_dataloader, desc=f'Epoch {epoch}') as pbar:
        for step, data_item in enumerate(pbar):
            optimizer.zero_grad()
            
            image = data_item['image'].to(device)
            label = data_item['label'].to(device)
            
            loss = model(
                image=image,  # Use keyword arguments
                train_labels=label[:, 0, :, :, :],
                train_organs=None
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Record loss for each step
            if local_rank == 0:  # Only record on the main process
                global_step = epoch * len(train_dataloader) + step
                writer.add_scalar('train/step_Loss', loss.item(), global_step)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                            'avg_loss': f'{total_loss/(step+1):.4f}'})
    
    return total_loss / len(train_dataloader)


if __name__ == '__main__':
    # set base parameters
    batch_size = 4
    num_workers = 4
    num_epochs = 3000
    initial_lr = 1e-5
    eta_min = 1e-6
    train_root_path = '/hpc2hdd/home/ydu709/data/3D_train_npz_random_10percent_16G'
    resume_checkpoint = './epoch_2000_loss_0.2232.pth'
    # resume_checkpoint = './SegVol_v1.pth'
    save_dir = './ckpts_fm3d_segvol'
    ###########################
    model_dir = './segvol'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.makedirs(save_dir, exist_ok=True)
    train_file_paths = glob.glob(os.path.join(train_root_path, '**', '*.npz'), recursive=True)

    # load model
    clip_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, test_mode=False)
    model = SegVolModel(config)
    model.model.text_encoder.tokenizer = clip_tokenizer
    # Save processor reference before wrapping with DDP
    processor = model.processor

    # set DDP
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    model.to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )
    if local_rank == 0:
        print('model load done')

    # Set optimizer
    start_epoch = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-5)
    # set your own scheduler
    # ...
    best_loss = float('inf')

    # load checkpoint
    start_epoch, best_loss = load_checkpoint(resume_checkpoint, device, model, optimizer, None, num_epochs, eta_min)

    # build dataset and dataloader
    train_dataset = SegDatasetTrain(train_file_paths, processor)
    print('train dataset size:', len(train_dataset))
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # 使用DistributedSampler时需要设为False
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=train_sampler,
        pin_memory=True
    )

    # set writer
    if local_rank == 0:
        writer = SummaryWriter(os.path.join(save_dir, 'logs'))
        if best_loss == float('inf'):  # If no checkpoint was restored
            best_loss = float('inf')
    else:
        writer = None
        best_loss = None
    
    # train loop
    print(f'start training from epoch {start_epoch}')
    for epoch in range(start_epoch, num_epochs):  # Start training from the restored epoch
        train_sampler.set_epoch(epoch)  # Ensure different data distribution for each epoch
        train_dataset.set_epoch(epoch)
        if local_rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current learning rate: {current_lr:.10f}')
        avg_loss = train_one_epoch(model, train_dataloader, optimizer, epoch)
        
        # Update learning rate
        # scheduler.step()
        if local_rank == 0:  # Only print and save on the main process
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Current learning rate: {current_lr:.10f}')
            
            # Record learning rate and average loss
            writer.add_scalar('train/Learning_Rate', current_lr, epoch)
            writer.add_scalar('train/avg_loss', avg_loss, epoch)
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),  # Note: use model.module here
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            
            # Periodically save model and validate
            if (epoch+1) % 25 == 0:
                ckpt_path = os.path.join(save_dir, f'epoch_{epoch+1}_loss_{avg_loss:.4f}.pth')
                torch.save(checkpoint, ckpt_path)
                print(f'save checkpoint to {ckpt_path}')
                # validation
                avg_bbox_dice = val_model(local_rank, model_dir, ckpt_path)
                writer.add_scalar('validation/BBox_Dice', avg_bbox_dice, epoch)
                print(f'Epoch {epoch} - Average BBox Dice: {avg_bbox_dice:.4f}')
            
            print(f'Epoch {epoch} - Average Loss: {avg_loss:.4f}')

    # Clean up process group and close writer
    if local_rank == 0:
        writer.close()
    dist.destroy_process_group()