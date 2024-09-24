import os
import os.path as osp
import sys
import cv2
import numpy as np
import torch

import warnings
import difflib
import SimpleITK as sitk
import scipy.spatial

from mmseg.apis import init_model, inference_model
from mmengine.utils import mkdir_or_exist
from medpy.io import load as loadnii
from medpy.io import save as savenii

name_t1_bet = str(sys.argv[1])
name_flr = str(sys.argv[2])
name_WMHs = str(sys.argv[3])
label_WMHs = str(sys.argv[4])
model_name = str(sys.argv[5])

#Data flow - read t1, FLAIR, Mask & output name.
#          - preprocess t1 & FLAIR.
#          - run remos with temp_data
#          - load segmentation
#          - create .nii file from segmentation (Optional)
#          - evaluate & save results

#Functions

#Crop from center
# def center_crop_batch(images, crop_size):
#     cropped_images = []
#     extra_info_list = []
#     images = np.transpose(images, (2, 0, 1))

#     for image in images:
#         height, width = image.shape[:2]
#         crop_height, crop_width = crop_size

#         # Calculate the starting point for the crop
#         start_x = max(0, int((width - crop_width) / 2))
#         start_y = max(0, int((height - crop_height) / 2))

#         # Calculate the ending point for the crop
#         end_x = min(width, start_x + crop_width)
#         end_y = min(height, start_y + crop_height)

#         # Calculate the border dimensions
#         border_left = int(max(0, crop_width / 2 - width) / 2)
#         border_right = int(max(0, crop_width / 2 - width) / 2)
#         border_top = int(max(0, crop_height - height) / 2)
#         border_bottom = int(max(0, crop_height - height) / 2)

#         # Perform the center crop
#         cropped = image[start_y:end_y, start_x:end_x]

#         # Expand the border if necessary
#         cropped = cv2.copyMakeBorder(cropped, border_top, border_bottom, border_left, border_right, cv2.BORDER_REPLICATE)

#         cropped_images.append(cropped)

#         # Store extra information about the original image size and crop dimensions
#         extra_info = {
#             'original_height': height,
#             'original_width': width,
#             'start_x': start_x,
#             'start_y': start_y,
#             'end_x': end_x,
#             'end_y': end_y,
#             'crop_height': crop_height,
#             'crop_width': crop_width,
#         }

#         extra_info_list.append(extra_info)

#     return np.array(cropped_images), extra_info_list


#Adaptable preprocessing for mri images

def insert_matrix_in_center(data, dims, ml=500, cs=112):
    x_s, y_s, z_s = dims
    
    min_x_axis = int(ml/2-x_s/2)
    max_x_axis = int(min_x_axis + x_s)
    min_y_axis = int(ml/2-y_s/2)
    max_y_axis = int(min_y_axis + y_s)

    T1_data_xyCtr = np.zeros((ml,ml,z_s))
    T1_data_xyCtr[min_x_axis:max_x_axis, min_y_axis:max_y_axis, :] = data
    T1_data_xyCtr = T1_data_xyCtr[int(ml/2-cs):int(ml/2+cs), int(ml/2-cs):int(ml/2+cs), :]
    
    T1_data_xyCtr  = np.transpose(T1_data_xyCtr , (2, 0, 1))
    
    info = min_x_axis, max_x_axis, min_y_axis, max_y_axis
    
    return T1_data_xyCtr, info

#Adaptable preprocessing for mri images
def preprocessing_mri(images, label=None):
  flair, t1 = images
  dim_info = [flair.shape[0], flair.shape[1], flair.shape[2]]
  
  #Crop images
  c_flair, info = insert_matrix_in_center(flair, dim_info)
  c_t1, _ = insert_matrix_in_center(t1, dim_info)
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
  new2_flair = z_score(new_flair)
  new2_t1 = z_score(new_t1)


  f_image = np.concatenate((new2_flair, new2_t1), 3)
#   new_img = t1t2(new2_flair, new2_t1,True)
#   new_img = np.where(new_t1 == 0, 0, new_img)
  new_img = np.zeros(new_flair.shape, dtype=np.float64)
  f_image = np.concatenate((f_image, new_img), 3)
  
  extra = [(np.min(sum_nozero), np.max(sum_nozero)), info]
  
  
#   f_image = np.concatenate((new2_flair, new2_flair), 3)
#   #new_img = t1t2(new_flair,new_t1,True)
#   new_img = np.zeros(new2_flair.shape, dtype=np.float64)
#   f_image = np.concatenate((f_image, new2_flair), 3)


  return f_image, extra

# #Create t1/t2 image
# def t1t2(a,b, truncate=False):
#     div = np.divide(a,b, where=b!=0)
#     if truncate:
#         percentile_1 = np.percentile(div, 1)
#         percentile_99 = np.percentile(div, 99)
#         div = np.clip(div,percentile_1,percentile_99)
#         normalized_data = (div - percentile_1) / (percentile_99 - percentile_1)
#         return normalized_data
#     return div

#minmax normalization
def minmax(image_data):
    percentile_1 = np.min(image_data)
    percentile_99 = np.max(image_data)
    normalized_data = (image_data - percentile_1) / (percentile_99 - percentile_1)
    return normalized_data

#Z score normalization
def z_score(data, lth = 0.02, uth = 0.98):
    
    temp = np.sort(data[data>0])
    lth_num = int(temp.shape[0]*0.05)
    uth_num = int(temp.shape[0]*0.95)
    data_mean = np.mean(temp[lth_num:uth_num])
    data_std = np.std(temp[lth_num:uth_num])
    data = (data - data_mean)/data_std
    
    return data


# data read
T1_bet_data, T1_bet_header = loadnii(name_t1_bet)
FLAIR_bet_data, FLAIR_header = loadnii(name_flr)

#Preprocessing
data, info = preprocessing_mri((FLAIR_bet_data, T1_bet_data))

#Iniciar modelo
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = init_model(f"configs/wmh_final/{model_name}.py", f"work_dirs/{model_name}/iter_80000.pth", "cuda:0")
    #model2 = init_model("configs/wmh_brats/remos_w3_try2.py", "work_dirs/remos_w3_try2/iter_80000.pth", "cuda:0")
    #model3 = init_model("configs/wmh_brats/remos_w3_try3.py", "work_dirs/remos_w3_try3/iter_80000.pth", "cuda:0")
index = 0
segmentation = []
segmentation_x = []
segmentation_y = []
segmentation_xy = []

data_x = np.flip(data.copy(), 1)
data_y = np.flip(data.copy(), 2)
data_xy = np.flip(np.flip(data.copy(), 1), 2)

for i in range(data.shape[0]):
    l_index = len(str(index))
    num = "0"*(6-l_index) + str(index)
    
    #Save images
    slice_data = np.swapaxes(data[i,:, :, :], 0, 1)
    slice_data_x = np.swapaxes(data_x[i,:, :, :], 0, 1)
    slice_data_y = np.swapaxes(data_y[i,:, :, :], 0, 1)
    slice_data_xy = np.swapaxes(data_xy[i,:, :, :], 0, 1)
    
    result = inference_model(model, slice_data).pred_sem_seg.values()[0]
    result_x = torch.flip(inference_model(model, slice_data_x).pred_sem_seg.values()[0], [2])
    result_y = torch.flip(inference_model(model, slice_data_y).pred_sem_seg.values()[0], [1])
    result_xy = torch.flip(inference_model(model, slice_data_xy).pred_sem_seg.values()[0], [1, 2])
    
    # results_FE = (result + result_x + result_y + result_xy)/4.0
    # final_results = (results_FE >= 0.5)
    # slice_data[:,:,2] = result
    # result = inference_model(model2, slice_data).pred_sem_seg.values()[0].cpu()
    # slice_data[:,:,2] = result
    # result = inference_model(model3, slice_data)
    
    segmentation.append(result)
    segmentation_y.append(result_y)
    segmentation_x.append(result_x)
    segmentation_xy.append(result_xy)
    #tifffile.imwrite('{}/imgs/test/{}.tiff'.format(out_dir,num), slice_data)
    
    index +=1
    
#final_segmentation = (torch.cat(segmentation, dim=0) + torch.cat(segmentation_x, dim=0))/2
final_segmentation = (torch.cat(segmentation, dim=0) + torch.cat(segmentation_x, dim=0) + torch.cat(segmentation_y, dim=0) + torch.cat(segmentation_xy, dim=0))/4
final_segmentation = (final_segmentation >= 0.2)
#Save File 
x_s, y_s, z_s = FLAIR_bet_data.shape[0], FLAIR_bet_data.shape[1], FLAIR_bet_data.shape[2]


#final_segmentation = torch.cat(segmentation, dim=0)
final_segmentation = final_segmentation.cpu()
final_segmentation = np.swapaxes(final_segmentation, 0, 2)

min_x_axis = info[1][0]
max_x_axis = info[1][1]
min_y_axis = info[1][2]
max_y_axis = info[1][3]


ml = 500
cs = 112
seg_ensemble_WMHs = np.zeros((ml,ml,z_s))
seg_ensemble_WMHs[int(ml/2-cs):int(ml/2+cs), int(ml/2-cs):int(ml/2+cs), info[0][0]:info[0][1]+1] = final_segmentation
seg_ensemble_WMHs = seg_ensemble_WMHs[min_x_axis:max_x_axis, min_y_axis:max_y_axis, :]

# ml = 500
# cs = 112
# seg_ensemble_WMHs = np.zeros((ml, ml, z_s))
# seg_ensemble_WMHs[int(ml/2-cs):int(ml/2+cs), int(ml/2-cs):int(ml/2+cs), info[0][0]:info[0][1]+1] = final_segmentation
# seg_ensemble_WMHs = seg_ensemble_WMHs[int(ml/2-x_s/2):int(ml/2+x_s/2), int(ml/2-y_s/2):int(ml/2+y_s/2), :]


#Uncomment for save
# savenii(seg_ensemble_WMHs, name_WMHs, FLAIR_header)

seg_ensemble_WMHs = np.swapaxes(seg_ensemble_WMHs, 0, 2)


""" Evaluación con métricas de WMHSC """

testDir        = label_WMHs # For example: '/data/Utrecht/0'
participantDir = 'tools_wmh/output' # For example: '/output/teamname/0'


def getImages(testFilename, resultFile):
    """Return the test and result images, thresholded and non-WMH masked."""
    testImage   = sitk.ReadImage(testFilename)
    resultImage = sitk.GetImageFromArray(resultFile)
    
    # Check for equality
    assert testImage.GetSize() == resultImage.GetSize(), "test_size = {}, result_size = {}".format(testImage.GetSize(), resultImage.GetSize())
    
    # Get meta data from the test-image, needed for some sitk methods that check this
    resultImage.CopyInformation(testImage)
    
    # Remove non-WMH from the test and result images, since we don't evaluate on that
    maskedTestImage = sitk.BinaryThreshold(testImage, 0.5,  1.5, 1, 0) # WMH == 1    
    nonWMHImage     = sitk.BinaryThreshold(testImage, 1.5,  2.5, 0, 1) # non-WMH == 2
    maskedResultImage = sitk.Mask(resultImage, nonWMHImage)
    
    # Convert to binary mask
    if 'integer' in maskedResultImage.GetPixelIDTypeAsString():
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 1, 1000, 1, 0)
    else:
        bResultImage = sitk.BinaryThreshold(maskedResultImage, 0.5, 1000, 1, 0)
        
    return maskedTestImage, bResultImage
    
    
def getDSC(testImage, resultImage):    
    """Compute the Dice Similarity Coefficient."""
    testArray   = sitk.GetArrayFromImage(testImage).flatten()
    resultArray = sitk.GetArrayFromImage(resultImage).flatten()
    
    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray) 
    

def getHausdorff(testImage, resultImage):
    """Compute the Hausdorff distance."""
    
    # Hausdorff distance is only defined when something is detected
    resultStatistics = sitk.StatisticsImageFilter()
    resultStatistics.Execute(resultImage)
    if resultStatistics.GetSum() == 0:
        return float('nan')
        
    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eTestImage   = sitk.BinaryErode(testImage, (1,1,0) )
    eResultImage = sitk.BinaryErode(resultImage, (1,1,0) )
    
    hTestImage   = sitk.Subtract(testImage, eTestImage)
    hResultImage = sitk.Subtract(resultImage, eResultImage)    
    
    hTestArray   = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)   
        
    # Convert voxel location to world coordinates. Use the coordinate system of the test image
    # np.nonzero   = elements of the boundary in numpy order (zyx)
    # np.flipud    = elements in xyz order
    # np.transpose = create tuples (x,y,z)
    # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
    testCoordinates   = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hTestArray) ))]
    resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in np.transpose( np.flipud( np.nonzero(hResultArray) ))]
        
            
    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):    
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]
    
    # Compute distances from test to result; and result to test
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)    
    
    return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))
    
    
def getLesionDetection(testImage, resultImage):    
    """Lesion detection metrics, both recall and F1."""
    
    # Connected components will give the background label 0, so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()    
    ccFilter.SetFullyConnected(True)
    
    # Connected components on the test image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccTest = ccFilter.Execute(testImage)    
    lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))
    
    ccTestArray = sitk.GetArrayFromImage(ccTest)
    lResultArray = sitk.GetArrayFromImage(lResult)
    
    # recall = (number of detected WMH) / (number of true WMH) 
    nWMH = len(np.unique(ccTestArray)) - 1
    if nWMH == 0:
        recall = 1.0
    else:
        recall = float(len(np.unique(lResultArray)) - 1) / nWMH
    
    # Connected components of results, to determine number of detected lesions
    ccResult = ccFilter.Execute(resultImage)
    lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))
    
    ccResultArray = sitk.GetArrayFromImage(ccResult)
    lTestArray = sitk.GetArrayFromImage(lTest)
    
    # precision = (number of detections that intersect with WMH) / (number of all detections)
    nDetections = len(np.unique(ccResultArray)) - 1
    if nDetections == 0:
        precision = 1.0
    else:
        precision = float(len(np.unique(lTestArray)) - 1) / nDetections
    
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * (precision * recall) / (precision + recall)
    
    return recall, f1    

    
def getAVD(testImage, resultImage):   
    """Volume statistics."""
    # Compute statistics of both images
    testStatistics   = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()
    
    testStatistics.Execute(testImage)
    resultStatistics.Execute(resultImage)
        
    return float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100


"""Main function"""
    
testImage, resultImage = getImages(testDir, seg_ensemble_WMHs)

dsc = getDSC(testImage, resultImage)
h95 = getHausdorff(testImage, resultImage)
avd = getAVD(testImage, resultImage)    
recall, f1 = getLesionDetection(testImage, resultImage)


results_txt = f'tools_wmh/output/resultados_{model_name}.txt'

data = open(results_txt, 'a')
data.write("{},{},{},{},{}\n".format(dsc, h95, avd, recall, f1))
data.close()