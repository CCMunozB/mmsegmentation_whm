import numpy as np
import tifffile
import os
import cv2
import nibabel as nib
import mmcv
import os.path as osp
from mmengine.utils import mkdir_or_exist


def insert_matrix_in_center(data, dims, ml=512, cs=256):
    x_s, y_s, z_s = dims
    
    min_x_axis = int(ml/2-x_s/2)
    max_x_axis = int(min_x_axis + x_s)
    min_y_axis = int(ml/2-y_s/2)
    max_y_axis = int(min_y_axis + y_s)

    T1_data_xyCtr = np.zeros((ml,ml,z_s))
    T1_data_xyCtr[min_x_axis:max_x_axis, min_y_axis:max_y_axis, :] = data
    T1_data_xyCtr = T1_data_xyCtr[int(ml/2-cs):int(ml/2+cs), int(ml/2-cs):int(ml/2+cs), :]
    
    T1_data_xyCtr  = np.transpose(T1_data_xyCtr , (2, 0, 1))
    
    return T1_data_xyCtr

def preprocessing_mri(images, label, val):
  flair, t1 = images
  dim_info = [flair.shape[0], flair.shape[1], flair.shape[2]]
  
  #Crop images
  c_flair = insert_matrix_in_center(flair, dim_info)
  c_t1 = insert_matrix_in_center(t1, dim_info)
  c_label = insert_matrix_in_center(label, dim_info)

  #Erase no info images
  if val:
    sum_nozero = np.where(np.sum(c_flair, axis=(1, 2)) > 0)
  else:
    sum_nozero = np.where(np.sum(c_label, axis=(1, 2)) > 0)
    
  new_flair = c_flair[sum_nozero,:,:]
  new_flair = new_flair.astype(np.float64)

  new_t1 = c_t1[sum_nozero,:,:]
  new_t1 = new_t1.astype(np.float64)

  new_label = c_label[sum_nozero,:,:]
  new_label[new_label != 1] = 0
  new_label = new_label.astype(np.uint8)

  new2_flair = z_score(new_flair)
  new2_t1 = z_score(new_t1)

  f_image = np.concatenate((new2_flair, new2_t1), 0)
  #new_img = t1t2(new2_flair, new2_t1,True)
  #new_img = np.where(new_t1 == 0, 0, new_img)
  new_img = np.zeros(new_flair.shape, dtype=np.float64)
  f_image = np.concatenate((f_image, new_img), 0)
  
  
#   f_image = np.concatenate((new2_flair, new2_flair), 3)
#   new_img = np.zeros(new2_flair.shape, dtype=np.float64)
#   f_image = np.concatenate((f_image, new2_flair), 3)


  return f_image, new_label

def t1t2(a,b, truncate=False):
    div = np.divide(a,b, where=b!=0)
    if truncate:
        percentile_1 = np.percentile(div, 1)
        percentile_99 = np.percentile(div, 99)
        div = np.clip(div,percentile_1,percentile_99)
        normalized_data = (div - percentile_1) / (percentile_99 - percentile_1)
        return normalized_data
    return div

def z_score(data, lth = 0.02, uth = 0.98):
    
    temp = np.sort(data[data>0])
    lth_num = int(temp.shape[0]*0.02)
    uth_num = int(temp.shape[0]*0.98)
    data_mean = np.mean(temp[lth_num:uth_num])
    data_std = np.std(temp[lth_num:uth_num])
    data = (data - data_mean)/data_std
    
    return data

def do(root):
    root_path = f"{root}"
    dirs = os.listdir(root_path + "/FLAIR")
    np.random.seed(95)
    # np.random.shuffle(dirs)
    # l = len(dirs)
    # train_dirs = dirs[:int(l*0.8)]
    # test_file = dirs[int(l*0.8):]
    # l = len(test_file)
    # val_file = test_file[int(l*0.5):]
    
    train_dirs = [0, 11, 17, 19, 2, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 4, 41, 49, 6, 8, #Ultrecht
                  50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, #Singapore
                  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 126, 132, 137, 144] #Amsterdam
    
    #Select 20% for Validation & Test for each type
    #v0 = np.random.choice(range(0,49),12, replace=False)
    #v1 = np.random.choice(range(50,99),12, replace=False)
    #v2 = np.random.choice(range(100,149),12, replace=False)
    #v3 = np.random.choice(range(150,159),2, replace=False)
    #v4 = np.random.choice(range(160,169),2, replace=False)
    
    #test_file = []
    #val_file = []
    # for vector in [v0, v1, v2]:
        # test_file = np.concatenate((test_file,vector[int(len(vector)/2):]))
        # val_file = np.concatenate((val_file,vector[:int(len(vector)/2)]))
    
    
    out_dir = 'data/WMH'
    
    print('Making directories...')
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, 'imgs'))
    mkdir_or_exist(osp.join(out_dir, 'imgs', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'imgs', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'label'))
    mkdir_or_exist(osp.join(out_dir, 'label', 'train'))
    mkdir_or_exist(osp.join(out_dir, 'label', 'val'))
    
    
    index = 0
    output_info = open("info_test.txt", 'w')
    for file in range(0,170):
        l = len(str(file))
        filename = str(file) + ".nii.gz" if l == 3 else "0"*(3-l) + str(file) + ".nii.gz"
        #Obtain data
        data_flair = nib.load(root_path + f"/FLAIR/{filename}" ).get_fdata()
        data_t1 = nib.load(root_path + f"/T1/{filename}" ).get_fdata()
        data_label = nib.load(root_path + f"/WMH/{filename}" ).get_fdata()
        
        #Preprocessing
        i_index= index
        
        if file in train_dirs:
            data = preprocessing_mri((data_flair, data_t1), data_label, val=True)
            for i in range(data[0].shape[1]):
                l_index = len(str(index))
                num = "0"*(6-l_index) + str(index)
                
                #Save images
                slice_data = data[0][:,i, :, :]
                slice_data = np.float32(np.swapaxes(slice_data, 0, -1))
                tifffile.imwrite('{}/imgs/train/{}.tiff'.format(out_dir,num), slice_data)
                
                #Save labels
                slice_data = data[1][:,i,:, :]
                slice_data = np.swapaxes(slice_data, 0, -1)
                #Dilatation
                #slice_data = scipy.ndimage.binary_dilation(slice_data).astype(np.int8)
                mmcv.imwrite(slice_data, '{}/label/train/{}.png'.format(out_dir,num))
                
                index +=1
                print(index, end="\r")
        else:
            data = preprocessing_mri((data_flair, data_t1), data_label, val=True)
            for i in range(data[0].shape[1]):
                l_index = len(str(index))
                num = "0"*(6-l_index) + str(index)
                
                #Save images
                slice_data = data[0][:,i, :, :]
                slice_data = np.float32(np.swapaxes(slice_data, 0, -1))
                tifffile.imwrite('{}/imgs/val/{}.tiff'.format(out_dir,num), slice_data)
                
                #Save labels
                slice_data = data[1][:,i,:, :]
                slice_data = np.swapaxes(slice_data, 0, -1)
                #Dilatation
                #slice_data = scipy.ndimage.binary_dilation(slice_data).astype(np.int8)
                mmcv.imwrite(slice_data, '{}/label/val/{}.png'.format(out_dir,num))
                
                index +=1
                print(index, end="\r")
        
        output_info.write("{}\t{}\t{}\n".format(filename, i_index, index))
            





if __name__=='__main__':
    do("/home/electroscian/ownCloud/Competencia/Datos_ROBEX")
    