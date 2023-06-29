import numpy as np
import tifffile
import os


def change(dir):
    lista = np.sort(os.listdir(dir))
    for val in lista:
        data = tifffile.imread(dir+ val)
        im = np.invert(data)
        tifffile.imwrite(dir + val, im)
        print(val, end="\r")

if __name__=='__main__':
    change("./data/train/label/")
    change("./data/val/label/")
    