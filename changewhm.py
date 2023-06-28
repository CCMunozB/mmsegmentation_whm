import cv2
import numpy as np
import tifffile
import os

train_label_path = "./data/train/label/"
val_label_path = "./data/val/label/"

for n,val in enumerate(os.listdir(val_label_path)):
    data = cv2.imread(val_label_path + val)
    data[data == 233] = 1
    tifffile.imwrite(val_label_path + val, data[:,:,0])
    print(n, end="\r")

for n,val in enumerate(os.listdir(train_label_path)):
    data = cv2.imread(train_label_path + val)
    data[data == 233] = 1
    tifffile.imwrite(train_label_path + val, data[:,:,0])
    print(n, end="\r")