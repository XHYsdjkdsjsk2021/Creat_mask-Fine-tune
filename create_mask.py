import cv2
import os
import numpy as np
import random
from PIL import Image

def rgb2gray(rgb_img):
    gray = rgb_img[:, :, 0] * 0.299 + rgb_img[:, :, 1] * 0.587 + rgb_img[:, :, 2] * 0.114
    return gray


def im_binary(gray_image, t=130):
    binary_image = np.zeros(shape=(gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            if gray_image[i][j] > t:
                binary_image[i][j] = 255
            else:
                binary_image[i][j] = 0
    return binary_image

def img_erode(bin_im, kernel, center_coo):
    kernel_w = kernel.shape[0]
    kernel_h = kernel.shape[1]
    if kernel[center_coo[0], center_coo[1]] == 0:
        raise ValueError("指定原点不在结构元素内！")
    erode_img = np.zeros(shape=bin_im.shape)
    for i in range(center_coo[0], bin_im.shape[0]-kernel_w+center_coo[0]+1):
        for j in range(center_coo[1], bin_im.shape[1]-kernel_h+center_coo[1]+1):
            a = bin_im[i-center_coo[0]:i-center_coo[0]+kernel_w,
                j-center_coo[1]:j-center_coo[1]+kernel_h]  # 找到每次迭代中对应的目标图像小矩阵
            if np.sum(a * kernel) == np.sum(kernel):  # 判定是否“完全重合”
                erode_img[i, j] = 1
    return erode_img

def img_dilate(bin_im, kernel, center_coo):
    kernel_w = kernel.shape[0]
    kernel_h = kernel.shape[1]
    if kernel[center_coo[0], center_coo[1]] == 0:
        raise ValueError("指定原点不在结构元素内！")
    dilate_img = np.zeros(shape=bin_im.shape)
    for i in range(center_coo[0], bin_im.shape[0] - kernel_w + center_coo[0] + 1):
        for j in range(center_coo[1], bin_im.shape[1] - kernel_h + center_coo[1] + 1):
            a = bin_im[i - center_coo[0]:i - center_coo[0] + kernel_w,
                j - center_coo[1]:j - center_coo[1] + kernel_h]
            dilate_img[i, j] = np.max(a * kernel)  # 若“有重合”，则点乘后最大值为0
    return dilate_img


def create_RGBA(path): #路径、透明度
    img_path = path
    center_coo=[0,0]
    kernel1 = np.ones(shape=(1, 1)) 
    image = Image.open(img_path)
    mask_ = Image.open('/data/haoyuan/sta/dongfangweishi.png').resize((100,100)) #190,70
    mask_array = np.array(mask_)
    gray = rgb2gray(mask_array)
    th_img = im_binary(gray)

    mask_re = img_dilate(th_img,kernel=kernel1,center_coo=center_coo)
    # mask_re = img_erode(close_img, kernel=kernel1,center_coo=center_coo)
    mask_re = Image.fromarray(mask_re)
    r,g,b,mask=mask_.split()
    w,h = image.size
    width,height = mask_.size
    x = random.randint(0,w-width)
    y = random.randint(0,h-height)
    image.paste(mask_,(x,y),mask=mask)
    # final_img = img[y:y+h_alp, x:x+w_alp]
 

    region = image.crop((x,y,x+width,y+height))
    # image.save("img120.png")
    
    return region

image_path = '/data/haoyuan/testttt/'
files = os.listdir(image_path)
num = 0
for file in files:
    num = num + 1
    p = image_path+file
    for x in range(2):
        final_image = create_RGBA(p)
        final_image.save('/data/haoyuan/maskdata/5/'+str(x)+'dongfangweishi_'+str(num)+'.jpg')
        print(p)

    