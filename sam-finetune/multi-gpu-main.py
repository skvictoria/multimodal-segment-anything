# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:04:42 2024
Revised on Mon May 13 2024

@author: Mohammed
@revised: Seulgi (Multi-GPU (DDP))
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
from skimage.transform import resize
import numpy as np
from torch.utils.data import Dataset
from transformers import SamProcessor
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader
from tqdm import tqdm
from statistics import mean
import torch
from torch.optim import Adam
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import logging
from PIL import Image


class SAMDataset(Dataset):
  def __init__(self, seismic_cube,label_cube, processor):
    self.seismic =seismic_cube
    self.label = label_cube
    self.processor = processor

    ########################## CHANGED FOR MULTI-GPU PROCESSING ########################################
    self.transform = Compose([
            Resize(256),  # Reduce image size
            ToTensor(),
        ])
    #####################################################################################################

  def __len__(self):
    return len(self.seismic)

  def __getitem__(self, idx):
    image = self.seismic[idx]
    ground_truth_mask = self.label[idx]

    ########################## CHANGED FOR MULTI-GPU PROCESSING ########################################
    if image.min() < 0:
        image = (image + 1) * 127.5
    image = image.astype('uint8') 
    image = Image.fromarray(image, 'RGB')
    image = self.transform(image)
    ground_truth_mask = torch.tensor(ground_truth_mask, dtype=torch.long).unsqueeze(0)
    inputs = self.processor(image.unsqueeze(0), input_labels=ground_truth_mask.unsqueeze(0), return_tensors="pt")
    #####################################################################################################

    inputs = {k:v.squeeze(0) for k,v in inputs.items()}
    return inputs

def scale_and_pad_image(image, new_max_length=1024):
    # Calculate the scaling factor
    height, width = image.shape[:2]
    scaling_factor = new_max_length / max(height, width)
    image=(image+1)/2
    # Scale the image
    new_height = int(height * scaling_factor)
    new_width = int(width * scaling_factor)
    scaled_image = resize(image, (new_height, new_width), preserve_range=True, anti_aliasing=True).astype(image.dtype)
    
    # Calculate padding
    pad_height = new_max_length - new_height
    pad_width = new_max_length - new_width
    
    # Apply padding to the far left side
    padded_image = np.pad(scaled_image, ((0,pad_height), 
                                          (0,pad_width)), 
                          'constant', constant_values=0)
    
    return padded_image

def scale_and_pad_label(image, new_max_length=256):
    # Calculate the scaling factor
    height, width = np.shape(image)
    scaling_factor = new_max_length / max(height, width)
    
    # Scale the image
    new_height = int(height * scaling_factor)
    new_width = int(width * scaling_factor)
    scaled_image = resize(image, (new_height, new_width), preserve_range=True, anti_aliasing=True).astype(image.dtype)
    
    # Calculate padding
    pad_height = new_max_length - new_height
    pad_width = new_max_length - new_width
    
    # Apply padding to the far left side
    padded_image = np.pad((scaled_image+1)/6, ((0,pad_height), 
                                          (0,pad_width)), 
                          'constant', constant_values=0)
    
    return padded_image

########################## ADDED FOR MULTI-GPU PROCESSING ########################################
def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',  # 'nccl' is recommended for GPUs
        init_method='env://',  # Assumes environment variables are set for master address and port
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    print(f"Rank {rank} is set to use GPU {torch.cuda.current_device()} with {torch.cuda.get_device_properties(rank).total_memory} bytes of memory")


def cleanup():
    dist.destroy_process_group()
#####################################################################################################


logging.basicConfig(filename='debug.log', level=logging.DEBUG)
try:
    if __name__ == "__main__":
        ########################## ADDED FOR MULTI-GPU PROCESSING ########################################
        # Multi-gpu setting
        logging.info("Starting the distributed training setup.")
        world_size = torch.cuda.device_count()
        rank = int(os.getenv('LOCAL_RANK', 0))
        setup(rank, world_size)
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

        # Model Loading per gpu rank
        model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(rank)
        model = DDP(model, device_ids=[rank])
        ####################################################################################################
        
        mask_folder = 'data/train/train_labels.npy'
        mask_files = np.load(mask_folder)

        processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        
        l=np.load('data/train/train_seismic.npy')

        ########################## CHANGED FOR MULTI-GPU PROCESSING ########################################
        samples = np.array([scale_and_pad_image(i, max(i.shape)) for i in l])
        samples = np.stack([samples, samples, samples], axis=-1)
        # samples=np.array([np.dstack((scale_and_pad_image(i,max(i.shape)),
        #                             scale_and_pad_image(i,max(i.shape)),
        #                             scale_and_pad_image(i,max(i.shape)))) for i in l])
        #####################################################################################################

        msks=[ scale_and_pad_label(i) for i in mask_files]
        
        for name, param in model.module.named_parameters():
            if name.startswith("prompt_encoder"):
                param.requires_grad = False
            else:
                param.requires_grad = True

        optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0)
        loss_fn = torch.nn.MSELoss()

        train_dataset = SAMDataset(samples, msks, processor=processor)

        ########################## ADDED FOR MULTI-GPU PROCESSING ########################################
        # train sampler for multi-gpu processing
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        #####################################################################################################

        train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler)
        
        num_epochs = 50
        losses=[]
        nimprove=0

        for epoch in range(num_epochs):
            ########################## ADDED FOR MULTI-GPU PROCESSING ########################################
            # train sampler for multi-gpu processing
            train_sampler.set_epoch(epoch)
            #####################################################################################################
            model.train()
            epoch_losses = []
            if nimprove > 4:
                print("no imrpove on last 5 epochs")
                break
            
            ########################## ADDED FOR MULTI-GPU PROCESSING ########################################
            for batch_idx, batch in enumerate(tqdm(train_loader, disable=rank != 0)):
            #####################################################################################################
                for n, value in model.module.image_encoder.named_parameters():
                    if "Adapter" in n:
                        value.requires_grad = True
                    else:
                        value.requires_grad = False
                
                pixel_values = batch["pixel_values"].to(rank)
                ground_truth_masks = batch["input_labels"].float().to(rank).squeeze(2)
                
                img_embs=model.module.image_encoder(pixel_values)
                sparse_embeddings, dense_embeddings = model.module.prompt_encoder(points=None, boxes=None, masks=None)
                low_res_masks,iou = model.module.mask_decoder(image_embeddings=img_embs,
                                                    image_pe = model.module.prompt_encoder.get_dense_pe(),
                                                    sparse_prompt_embeddings=sparse_embeddings,
                                                    dense_prompt_embeddings=dense_embeddings, multimask_output=False)
                optimizer.zero_grad()
                loss = loss_fn(low_res_masks,ground_truth_masks)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                del pixel_values, ground_truth_masks, img_embs, low_res_masks
                torch.cuda.empty_cache()  # Clear cache if running on GPUs

            ########################## ADDED FOR MULTI-GPU PROCESSING ########################################
            if rank == 0:
            #####################################################################################################
                print(f'EPOCH: {epoch}')
                print(f'Mean loss: {mean(epoch_losses)}')
                losses.append(mean(epoch_losses))
                if epoch>1 and np.round(losses[-1],3)>=np.round(losses[-2],3):
                    nimprove+=1

        cleanup()
except Exception as e:
    logging.error("An error occurred during setup :", exc_info=True)
    cleanup()