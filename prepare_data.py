import numpy as np
import tifffile
import os
import cv2
import nibabel as nib
import mmcv
import os.path as osp
from medpy.io import load as loadnii
from medpy.io import save as savenii
from mmengine.utils import mkdir_or_exist


def process_data(name_t1_bet, name_t1_bet_mask, name_flr, name_wmh):
    
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
    
    WMH_bet_data, WMH_bet_header = loadnii(name_wmh)

    #Preprocessing
    ml = 512 # max length
    cs = 256 # crop size; cs*2 = crop sizef
    
    axial_sum_nonzero = np.where(np.sum(FLAIR_bet_data, axis=(0, 1)) > 0)

    z_axis_min = np.min(axial_sum_nonzero)
    z_axis_max = np.max(axial_sum_nonzero)

    #
    T1_data_zcrop = T1_bet_data[:,:,z_axis_min:z_axis_max+1]
    FLAIR_data_zcrop = FLAIR_bet_data[:,:,z_axis_min:z_axis_max+1]
    WMH_data_zcrop = WMH_bet_data[:,:,z_axis_min:z_axis_max+1]

    #
    T1_data_zscore = z_score(T1_data_zcrop)
    FLAIR_data_zscore = z_score(FLAIR_data_zcrop)
    WMH_data_zcrop[WMH_data_zcrop != 1] = 0

    #
    x_s, y_s, z_s = FLAIR_data_zscore.shape[0], FLAIR_data_zscore.shape[1], FLAIR_data_zscore.shape[2]

    min_x_axis = int(ml/2-x_s/2)
    max_x_axis = int(min_x_axis + x_s)
    min_y_axis = int(ml/2-y_s/2)
    max_y_axis = int(min_y_axis + y_s)

    T1_data_xyCtr = np.zeros((ml,ml,z_s))
    T1_data_xyCtr[min_x_axis:max_x_axis, min_y_axis:max_y_axis, :] = T1_data_zscore
    T1_data_xyCtr = T1_data_xyCtr[int(ml/2-cs):int(ml/2+cs), int(ml/2-cs):int(ml/2+cs), :]

    FLAIR_data_xyCtr = np.zeros((ml,ml,z_s))
    FLAIR_data_xyCtr[min_x_axis:max_x_axis, min_y_axis:max_y_axis, :] = FLAIR_data_zscore
    FLAIR_data_xyCtr = FLAIR_data_xyCtr[int(ml/2-cs):int(ml/2+cs), int(ml/2-cs):int(ml/2+cs), :]
    
    WMH_data_xyCtr = np.zeros((ml,ml,z_s))
    WMH_data_xyCtr[min_x_axis:max_x_axis, min_y_axis:max_y_axis, :] = WMH_data_zcrop
    WMH_data_xyCtr = WMH_data_xyCtr[int(ml/2-cs):int(ml/2+cs), int(ml/2-cs):int(ml/2+cs), :]

    #
    T1_data_swp = np.swapaxes(T1_data_xyCtr, 0, 2)
    FLAIR_data_swp = np.swapaxes(FLAIR_data_xyCtr, 0, 2)

    #
    T1_data_rsp = np.reshape(T1_data_swp,(z_s, cs*2, cs*2, 1))
    FLAIR_data_rsp = np.reshape(FLAIR_data_swp,(z_s, cs*2, cs*2, 1))
    ZEROS_data_rsp = np.zeros(FLAIR_data_rsp.shape, dtype=np.float32)
                
    concat_data = np.concatenate((FLAIR_data_rsp, T1_data_rsp), axis=3)
    data = np.concatenate((concat_data, ZEROS_data_rsp), axis=3)
    
    return data

train_dirs = [0, 11, 17, 19, 2, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 4, 41, 49, 6, 8, #Ultrecht
                50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, #Singapore
                100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 126, 132, 137, 144] #Amsterdam

def do(root):
    
    #Set root path
    root_path = f"{root}"
    dirs = os.listdir(root_path + "/FLAIR")
    
    #Create Directories
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
            data = process_data((data_flair, data_t1), data_label, val=True)
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
            data = process_data((data_flair, data_t1), data_label, val=True)
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
    

    