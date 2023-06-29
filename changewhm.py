import numpy as np
import tifffile
import os

train_label_path = "./data/train/imgs/"
val_label_path = "./data/val/imgs/"

for n,val in enumerate(os.listdir(val_label_path)):
    data = tifffile.imread(val_label_path + val)
    data = data[:,:,0:2]
    tifffile.imwrite(val_label_path + val, data)
    print(n, end="\r")

for n,val in enumerate(os.listdir(train_label_path)):
    data = tifffile.imread(train_label_path + val)
    data = data[:,:,0:2]
    tifffile.imwrite(train_label_path + val, data)
    print(n, end="\r")