import numpy as np
import cv2
import os


def change(dir):
    lista = np.sort(os.listdir(dir))
    for val in lista:
        data = cv2.imread(dir+ val)
        im = np.invert(data)
        cv2.imwrite(dir + val, im)
        print(val, end="\r")

if __name__=='__main__':
    change("./data/train/label/")
    change("./data/val/label/")
    