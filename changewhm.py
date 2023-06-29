import numpy as np
import tifffile
import os

def norm(img):
    norm_img = (img- np.min(img))/(np.max(img)- np.min(img))
    return norm_img

def norm2(img):
    down = np.percentile(img,2.0)
    up = np.percentile(img, 98.0)
    res = np.clip(img,down,up)
    a = res - down
    b = up - down
    f_res = np.divide(a,b, where=b!=0)
    f_res = np.expand_dims(f_res , 2)
    return f_res


def change(dir):
    lista = np.sort(os.listdir(dir))
    for val in lista:
        data = tifffile.imread(dir+ val)
        a = norm(data[:,:,0])
        b = norm(data[:,:,1])
        div = np.divide(a,b, where=b!=0)
        res = norm2(div)
        im = np.concatenate((data[:,:,0:2],res), axis=2)
        tifffile.imwrite(dir + val, im)
        print(val, end="\r")

if __name__=='__main__':
    change("./data/train/imgs/")
    change("./data/val/imgs/")
    