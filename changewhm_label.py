import numpy as np
import tifffile
import os


def change(dir):
    lista = np.sort(os.listdir(dir))
    for val in lista:
        data = tifffile.imread(dir+ val)
        im = np.invert(data)
        data = np.expand_dims(data, 2)
        im = np.expand_dims(im, 2)
        final = np.concatenate((data,im), axis=2)
        tifffile.imwrite(dir + val, final)
        print(val, end="\r")

if __name__=='__main__':
    change("./data/train/label/")
    change("./data/val/label/")
    