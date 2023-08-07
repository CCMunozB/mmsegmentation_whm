import numpy as np
import tifffile
import os
import cv2
import nibabel as nib
import mmcv
import os.path as osp
from mmengine.utils import mkdir_or_exist


def center_crop_batch(images, crop_size):
    cropped_images = []
    images = np.transpose(images, (2, 0, 1))

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
        
    return np.array(cropped_images)

def preprocessing_mri(images, label):
  flair, t1 = images

  #Crop images
  c_flair = center_crop_batch(flair, (224,224))
  c_t1 = center_crop_batch(t1, (224,224))
  c_label = center_crop_batch(label, (224,224))

  #Erase no info images
  sum_nozero = np.where(np.sum(c_flair, axis=(1, 2)) > 0)
  new_flair = c_flair[np.min(sum_nozero):np.max(sum_nozero)+1,:,:]
  new_flair = new_flair.astype(np.float64)
  new_flair = np.expand_dims(new_flair, 3)

  sum_nozero = np.where(np.sum(c_t1, axis=(1, 2)) > 0)
  new_t1 = c_t1[np.min(sum_nozero):np.max(sum_nozero)+1,:,:]
  new_t1 = new_t1.astype(np.float64)
  new_t1 = np.expand_dims(new_t1, 3)

  new_label = c_label[np.min(sum_nozero):np.max(sum_nozero)+1,:,:]
  new_label[new_label != 1] = 0
  new_label = new_label.astype(np.uint8)

  new_flair = minmax(new_flair)
  new_t1 = minmax(new_t1)


  f_image = np.concatenate((new_flair, new_t1), 3)
  #new_img = t1t2(new_flair,new_t1,True)
  new_img = np.zeros(new_flair.shape, dtype=np.float64)
  f_image = np.concatenate((f_image, new_img), 3)


  return f_image, new_label

def t1t2(a,b, truncate=False):
    div = np.divide(a,b, where=b!=0)
    if truncate:
        percentile_1 = np.percentile(div, 2)
        percentile_99 = np.percentile(div, 98)
        div = np.clip(div,percentile_1,percentile_99)
        normalized_data = (div - percentile_1) / (percentile_99 - percentile_1)
        return normalized_data
    return div

def minmax(image_data):
    percentile_1 = np.min(image_data)
    percentile_99 = np.max(image_data)
    normalized_data = (image_data - percentile_1) / (percentile_99 - percentile_1)
    return normalized_data

def do(root):
    root_path = f"{root}"
    dirs = os.listdir(root_path + "/FLAIR")
    np.random.seed(33)
    np.random.shuffle(dirs)
    l = len(dirs)
    train_dirs = dirs[:int(l*0.9)]
    val_dirs = dirs[int(l*0.9):]
    
    out_dir = 'data/WMH2'
    
    print('Making directories...')
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, 'imgs'))
    mkdir_or_exist(osp.join(out_dir, 'imgs', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'imgs', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'label'))
    mkdir_or_exist(osp.join(out_dir, 'label', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'label', 'val'))
    
    train_dims = []
    val_dims = []
    index = 0
    for file in train_dirs:
        #Obtain data
        data_flair = nib.load(root_path + f"/FLAIR/{file}" ).get_fdata()
        data_t1 = nib.load(root_path + f"/T1/{file}" ).get_fdata()
        data_label = nib.load(root_path + f"/WMH/{file}" ).get_fdata()
        
        #Preprocessing
        data = preprocessing_mri((data_flair, data_t1), data_label)
        train_dims.append(data[0].shape[0])
        
        for i in range(data[0].shape[0]):
            l_index = len(str(index))
            num = "0"*(6-l_index) + str(index)
            
            #Save images
            slice_data = data[0][i,:, :, :]
            tifffile.imwrite('data/WMH2/imgs/train/{}.tiff'.format(num), slice_data)
            
            #Save labels
            slice_data = data[1][i,:, :]
            #Dilatation
            #slice_data = scipy.ndimage.binary_dilation(slice_data).astype(np.int8)
            mmcv.imwrite(slice_data, 'data/WMH2/label/train/{}.png'.format(num))
            
            index +=1
            
            print(index, end="\r")
            
    for file in val_dirs:
        #Obtain data
        data_flair = nib.load(root_path + f"/FLAIR/{file}" ).get_fdata()
        data_t1 = nib.load(root_path + f"/T1/{file}" ).get_fdata()
        data_label = nib.load(root_path + f"/WMH/{file}" ).get_fdata()
        
        #Preprocessing
        data = preprocessing_mri((data_flair, data_t1), data_label)
        val_dims.append(data[0].shape[0])
        
        for i in range(data[0].shape[0]):
            l_index = len(str(index))
            num = "0"*(6-l_index) + str(index)
            
            #Save images
            slice_data = data[0][i,:, :, :]
            tifffile.imwrite('data/WMH2/imgs/val/{}.tiff'.format(num), slice_data)
            
            #Save labels
            slice_data = data[1][i,:, :]
            mmcv.imwrite(slice_data, 'data/WMH2/label/val/{}.png'.format(num))
            
            index +=1
            print(index, end="\r")





if __name__=='__main__':
    do("/home/electroscian/Downloads/Datos_ROBEX")
    