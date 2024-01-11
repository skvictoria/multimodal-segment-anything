import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def create_mask_image(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    return mask_image

if __name__=="__main__":
    sam_checkpoint = "notebooks/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    #initialize sam predictor
    predictor = SamPredictor(sam)

    color = np.array([30/255, 144/255, 255/255, 0.6])

    # image = cv2.imread('notebooks/images/000048.png')
    for imagefile in os.listdir('notebooks/images'):
        save_name = imagefile.replace('.','')
        image = cv2.imread('notebooks/images/'+imagefile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # sampredictor remembers the embedding
        predictor.set_image(image)

        input_point = np.array([[460, 195], [470, 195]])
        input_label = np.array([1, 1])
        
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )

        combined_image = image.copy()
        alpha_mask = masks.reshape(masks.shape[-2:]) * 0.6  # Adjust alpha value as needed
        for c in range(3):  # Assuming image is in RGB format
            combined_image[:,:,c] = np.where(masks, combined_image[:,:,c] * (1 - alpha_mask) + color[c] * 255 * alpha_mask, combined_image[:,:,c])
        plt.imsave(f"notebooks/save/{save_name}.png", combined_image)