import os
import sys
import ast
from PIL import Image
from numpy import array, stack, swapaxes, zeros
from medpy.io import load as loadnii
from medpy.io import save as savenii


def read_files(path, patient):
    """ Input path to read png segmentation files.
    Args:
        path (str): path to the folder containing the segmentation files
        start (int): start index of the files to read
        end (int): end index of the files to read
    Returns:
        list: list of segmentation files
    """
    
    with open("info_test.txt", "r") as info_file:
        for line in info_file:
            if line.startswith(patient):
                start = int(line.split("\t")[1])
                end = int(line.split("\t")[2])  
                break 
    files = []
    for i in range(start, end):
        filename = (6 - len(str(i)))*"0" + str(i)
        file = os.path.join(path, f"{filename}.png")
        files.append(file)
        
    return files

def create_nii_file(files, output_path, patient):
    """ Create a nii file from a list of segmentation files.
    Args:
        files (list): list of segmentation files
    Returns:
        nii file
    """
    image_stack = []
    for file in files:
        image = Image.open(file)
        image_array = array(image)
        image_stack.append(image_array)
        
    image_stack = stack(image_stack, axis=-1)
    image_stack = swapaxes(image_stack, 0, 1)
    
    final_seg = recover_image(image_stack, patient)
    
    savenii(final_seg, output_path, hdr=None)
    
def recover_image(segmentation, patient):
    with open("preprocess_info.txt", "r") as info_file:
        for line in info_file:
            if line.startswith(patient):
                info = ast.literal_eval(line.split("\t")[1])
                z_s = int(line.split("\t")[2])
                break 
            
    min_x_axis = info[1][0]
    max_x_axis = info[1][1]
    min_y_axis = info[1][2]
    max_y_axis = info[1][3]

    ml = 500
    cs = 112
    seg_ensemble_WMHs = zeros((ml,ml,z_s))
    seg_ensemble_WMHs[int(ml/2-cs):int(ml/2+cs), int(ml/2-cs):int(ml/2+cs), info[0][0]:info[0][1]+1] = segmentation
    seg_ensemble_WMHs = seg_ensemble_WMHs[min_x_axis:max_x_axis, min_y_axis:max_y_axis, :]
    
    return seg_ensemble_WMHs
    
if __name__ == "__main__":
    
    seg_path = str(sys.argv[1])
    patient = str(sys.argv[2])
    output_path = str(sys.argv[3])
    
    files = read_files(seg_path, patient)
    create_nii_file(files, output_path, patient)