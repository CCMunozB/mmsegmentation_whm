import os
import sys
from PIL import Image
from numpy import array, stack, eye
from medpy.io import load as loadnii
from medpy.io import save as savenii


def read_files(path, start, end):
    """ Input path to read png segmentation files.
    Args:
        path (str): path to the folder containing the segmentation files
        start (int): start index of the files to read
        end (int): end index of the files to read
    Returns:
        list: list of segmentation files
    """
    files = []
    for i in range(start, end):
        filename = (6 - len(str(i)))*"0" + str(i)
        file = os.path.join(path, f"{filename}.png")
        files.append(file)
        
    return files

def create_nii_file(files, output_path):
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
    
    savenii(image_stack, output_path, affine=eye(4))
    
if __name__ == "__main__":
    
    seg_path = str(sys.argv[0])
    seg_start = str(sys.argv[1]) 
    seg_end = str(sys.argv[2])
    output_path = str(sys.argv[3])
    
    files = read_files(seg_path, seg_start, seg_end)
    create_nii_file(files, output_path)