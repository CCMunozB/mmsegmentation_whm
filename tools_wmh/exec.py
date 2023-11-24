import os
import os.path as osp
import subprocess
import sys
import tifffile
import cv2
import mmcv
import numpy as np
import torch

from mmseg.apis import init_model, inference_model
from mmengine.utils import mkdir_or_exist
from medpy.io import load as loadnii
from medpy.io import save as savenii

name_t1_bet = str(sys.argv[1])
name_t1_bet_mask = str(sys.argv[2])
name_flr = str(sys.argv[3])
name_WMHs = str(sys.argv[4])

#Data flow - read t1, FLAIR, Mask & output name.
#          - preprocess t1 & FLAIR.
#          - run remos with temp_data
#          - load segmentation
#          - create .nii file from segmentation

#Functions

#Crop from center
def center_crop_batch(images, crop_size):
    cropped_images = []
    images = np.swapaxes(images, 2, 0)

    for image in images:
        height, width = image.shape[:2]
        crop_height, crop_width = crop_size
        
        # Calculate the starting point for the crop
        start_x = max(0, int((width - crop_width) / 2))
        start_y = max(0, int((height - crop_height) / 2))
        
        # Calculate the ending point for the crop
        end_x = min(width, start_x + crop_width)
        end_y = min(height, start_y + crop_height)
        
        # Calculate the border dimensions
        border_left = int(max(0, crop_width/2 - width)/2)
        border_right = int(max(0, crop_width/2 - width)/2)
        border_top = int(max(0, crop_height - height)/2)
        border_bottom = int(max(0, crop_height - height)/2)
        
        # Perform the center crop
        cropped = image[start_y:end_y, start_x:end_x]
        
        # Expand the border if necessary
        cropped = cv2.copyMakeBorder(cropped, border_top, border_bottom, border_left, border_right, cv2.BORDER_REPLICATE)
        
        cropped_images.append(cropped)
        
        extra = [start_x, start_y, end_x, end_y]
        
    return np.array(cropped_images), extra

#Adaptable preprocessing for mri images
def preprocessing_mri(images, label=None):
  flair, t1 = images

  #Crop images
  c_flair, crop_info = center_crop_batch(flair, (224,224))
  c_t1, _ = center_crop_batch(t1, (224,224))
  #c_label = center_crop_batch(label, (224,224))

  #Erase no info images
  sum_nozero = np.where(np.sum(c_flair, axis=(1, 2)) > 0)
  new_flair = c_flair[np.min(sum_nozero):np.max(sum_nozero)+1,:,:]
  new_flair = new_flair.astype(np.float64)
  new_flair = np.expand_dims(new_flair, 3)

  sum_nozero = np.where(np.sum(c_t1, axis=(1, 2)) > 0)
  new_t1 = c_t1[np.min(sum_nozero):np.max(sum_nozero)+1,:,:]
  new_t1 = new_t1.astype(np.float64)
  new_t1 = np.expand_dims(new_t1, 3)

#   new_label = c_label[np.min(sum_nozero):np.max(sum_nozero)+1,:,:]
#   new_label[new_label != 1] = 0
#   new_label = new_label.astype(np.uint8)
  
  #Test minmax and z score
  new_flair = z_score(new_flair)
  new_t1 = z_score(new_t1)


#   f_image = np.concatenate((new_flair, new_t1), 3)
#   #new_img = t1t2(new_flair,new_t1,True)
#   new_img = np.zeros(new_flair.shape, dtype=np.float64)
#   f_image = np.concatenate((f_image, new_img), 3)
  
  extra = [(np.min(sum_nozero), np.max(sum_nozero)), crop_info]
  
  
  f_image = np.concatenate((new_flair, new_flair), 3)
  #new_img = t1t2(new_flair,new_t1,True)
  new_img = np.zeros(new_flair.shape, dtype=np.float64)
  f_image = np.concatenate((f_image, new_flair), 3)


  return f_image, extra

#Create t1/t2 image
def t1t2(a,b, truncate=False):
    div = np.divide(a,b, where=b!=0)
    if truncate:
        percentile_1 = np.percentile(div, 2)
        percentile_99 = np.percentile(div, 98)
        div = np.clip(div,percentile_1,percentile_99)
        normalized_data = (div - percentile_1) / (percentile_99 - percentile_1)
        return normalized_data
    return div

#minmax normalization
def minmax(image_data):
    percentile_1 = np.min(image_data)
    percentile_99 = np.max(image_data)
    normalized_data = (image_data - percentile_1) / (percentile_99 - percentile_1)
    return normalized_data

#Z score normalization
def z_score(data, lth = 0.02, uth = 0.98):
    
    temp = np.sort(data[data>0])
    lth_num = int(temp.shape[0]*0.02)
    uth_num = int(temp.shape[0]*0.98)
    data_mean = np.mean(temp[lth_num:uth_num])
    data_std = np.std(temp[lth_num:uth_num])
    data = (data - data_mean)/data_std
    
    return data


# data read
T1_bet_data, T1_bet_header = loadnii(name_t1_bet)
T1_bet_mask_data, T1_bet_mask_header = loadnii(name_t1_bet_mask)

FLAIR_data, FLAIR_header = loadnii(name_flr)
FLAIR_bet_data = FLAIR_data*T1_bet_mask_data

#Preprocessing
data, info = preprocessing_mri((FLAIR_bet_data, T1_bet_data))

#Iniciar modelo
model = init_model("configs/whm/swin_upernet_test.py", "work_dirs/swin_upernet_whm_flair_cosh/iter_80000.pth", "cuda:0")

index = 0
segmentation = []
for i in range(data.shape[0]):
    l_index = len(str(index))
    num = "0"*(6-l_index) + str(index)
    
    #Save images
    slice_data = data[i,:, :, :]
    result = inference_model(model, slice_data)
    segmentation.append(result.pred_sem_seg.values()[0])
    #tifffile.imwrite('{}/imgs/test/{}.tiff'.format(out_dir,num), slice_data)
    
    index +=1
    print(index, end="\r")

#Save File 
x_s, y_s, z_s = FLAIR_data.shape[0], FLAIR_data.shape[1], FLAIR_data.shape[2]

final_segmentation = torch.cat(segmentation, dim=0)
final_segmentation = final_segmentation.cpu()
final_segmentation = np.swapaxes(final_segmentation, 0, 2)

seg_ensemble_WMHs = np.zeros((x_s, y_s, z_s))
seg_ensemble_WMHs[info[1][0]:info[1][2], info[1][1]:info[1][3], info[0][0]:info[0][1]+1] = final_segmentation

savenii(seg_ensemble_WMHs, name_WMHs, FLAIR_header)