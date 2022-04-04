import argparse
import os
import torch
import albumentations as A
import cv2
import glob
import torch.nn.functional as F
import time
from albumentations.pytorch import ToTensorV2
from efficient_model import Classifier



import imutils
import numpy as np
import pandas as pd


import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized



#파일내 이미지 하나 crop and resize
def crop_merge_one(img):
    #print("\ncrop_merge_one")
    #print(img)
    back_dir = './black.png'
    back = cv2.imread(back_dir)
    #print(back.shape)

    try:
        if img.endswith("yolo.png"):
            #name = img.split("/")[-1].split("-")[0]
            name = img[:-9]
            #print(name)

            # bounding box 위치 찾기
            image = cv2.imread(img)
            image = cv2.resize(image, dsize=(380, 380), interpolation=cv2.INTER_AREA)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_gray = cv2.resize(image_gray, dsize=(380, 380), interpolation=cv2.INTER_AREA)


            #cnts, _  = cv2.findContours(image_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #print("cnts",cnts)

            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            rect_img = cv2.inRange(image_hsv, (30, 100, 0), (90, 255, 255))
            cor = np.where(rect_img > 0 )
            #print(cor[0])
            #print(cor[1])
            cor0_set = set(cor[0])
            cor0_list = list(cor0_set)
            cor1_set = set(cor[1])
            cor1_list = list(cor1_set)
            #print(cor0_list)
            #print(cor1_list)

            # yolo bounding box 없으면
            if (len(cor0_list) ==0 & len(cor1_list)==0):
                back_resize = image
                cv2.imshow('crop', back_resize)
                cv2.imwrite(name + "-crop.png", back_resize)
                #print(name + "-crop.png")


            else:
                cor_list = []
                x = []
                for n in range(len(cor0_list)-1):
                    if cor0_list[n] + 2 > cor0_list[n + 1]:
                        x.append(cor0_list[n])
                    #print("x",x)
                    x1 = min(x)
                    x2 = max(x)

                y = []
                for m in range(len(cor1_list)-1):
                    if cor1_list[m] + 2 >  cor1_list[m+1] :
                        y.append(cor1_list[m])
                    #print("y",y)
                    y1 = min(y)
                    y2 = max(y)

                box_org = [x1,x2,y1,y2]
                #print("box_org",box_org)

                cor_list.append(box_org)
                #print("cor_list",cor_list)

                # roi = image[y: y + h, x: x + w]
                mask = cor_list[0]
                #print("mask", mask)
                x1 = mask[0]
                x2 = mask[1]
                y1 = mask[2]
                y2 = mask[3]
                w = mask[3] - mask[2]
                h = mask[1] - mask[0]
                #print("h", h)
                #print("w", w)


                # merge_img
                if h > w:
                    ori_img = name + ".png"
                    ori_img = cv2.imread(ori_img)
                    ori_img = cv2.resize(ori_img, dsize=(380, 380), interpolation=cv2.INTER_AREA)
                    #cv2.imshow('ori_img', ori_img)
                    #cv2.imshow('yolo_img', image)

                    back_resize = cv2.resize(back, dsize=(w, h), interpolation=cv2.INTER_AREA)
                    ori_imgcopy = ori_img.copy()
                    #print(back_resize.shape)
                    image[x1:x2, y1:y2] = back_resize
                    crop = cv2.subtract(ori_imgcopy, image)
                    cv2.imshow('crop', crop)
                    #cv2.waitKey(0)
                    cv2.imwrite(name + "-crop.png", crop)
                    #print(name + "-crop.png")

                elif h < w:
                    ori_img = name + ".png"
                    ori_img = cv2.imread(ori_img)
                    ori_img = cv2.resize(ori_img, dsize=(380, 380), interpolation=cv2.INTER_AREA)
                    #cv2.imshow('ori_img', ori_img)
                    #cv2.imshow('yolo_img', image)

                    back_resize = cv2.resize(back, dsize=(w, h), interpolation=cv2.INTER_AREA)
                    #print(back_resize.shape)
                    ori_imgcopy = ori_img.copy()
                    image[x1:x2, y1:y2] = back_resize
                    crop = cv2.subtract(ori_imgcopy, image)
                    cv2.imshow('crop', crop)
                    cv2.imwrite(name + "-crop.png", crop)
                    #print(name + "-crop.png")

                elif h == w:
                    ori_img = name + ".png"
                    ori_img = cv2.imread(ori_img)
                    ori_img = cv2.resize(ori_img, dsize=(380, 380), interpolation=cv2.INTER_AREA)
                    #cv2.imshow('ori_img', ori_img)
                    #cv2.imshow('yolo_img', image)

                    back_resize = cv2.resize(back, dsize=(h, h), interpolation=cv2.INTER_AREA)
                    # print(back_resize.shape)
                    ori_imgcopy = ori_img.copy()
                    image[x1:x2, y1:y2] = back_resize
                    crop = cv2.subtract(ori_imgcopy, image)
                    cv2.imshow('crop', crop)
                    cv2.imwrite(name + "-crop.png", crop)
                    #print(name + "-crop.png")


    except:
        pass

    return name









#파일내 모든 이미지 crop and resize
def crop_merge_all(yolo_source):
    back_dir = './black.png'
    back = cv2.imread(back_dir)
    #print(back.shape)

    #yolo_source = "./boundingbox"
    yolo_source_list = os.listdir(yolo_source)
    yolo_image_list = []
    for i in range(len(yolo_source_list)):
        data_image = yolo_source + "/" + yolo_source_list[i]  # 폴더
        #print("data: ",data_image)
        if data_image.endswith("yolo.png"):
            yolo_image_list.append(data_image)

    #print("\nyolo_image_list",yolo_image_list)
    for i in range(len(yolo_image_list)):
        # yolo 이미지
        img = yolo_image_list[i]
        #print("img",img)
        name = img.split("/")[-1].split("-")[0]

        # bounding box 위치 찾기
        image = cv2.imread(img)
        image = cv2.resize(image, dsize=(380, 380), interpolation=cv2.INTER_AREA)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = cv2.resize(image_gray, dsize=(380, 380), interpolation=cv2.INTER_AREA)


        #cnts, _  = cv2.findContours(image_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #print("cnts",cnts)

        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rect_img = cv2.inRange(image_hsv, (30, 100, 0), (90, 255, 255))
        cor = np.where(rect_img > 0 )
        #print(cor[0])
        #print(cor[1])
        cor0_set = set(cor[0])
        cor0_list = list(cor0_set)
        cor1_set = set(cor[1])
        cor1_list = list(cor1_set)
        #print(cor0_list)
        #print(cor1_list)

        # yolo bounding box 없으면
        if (len(cor0_list) ==0 & len(cor1_list)==0):
            back_resize = image
            cv2.imwrite(yolo_source + "/" + name + "-crop.png", back_resize)
            #print(yolo_source + "/" + name + "-crop.png")


        else:
            cor_list = []
            x = []
            for n in range(len(cor0_list)-1):
                if cor0_list[n] + 2 > cor0_list[n + 1]:
                    x.append(cor0_list[n])
                #print("x",x)
                x1 = min(x)
                x2 = max(x)

            y = []
            for m in range(len(cor1_list)-1):
                if cor1_list[m] + 2 >  cor1_list[m+1] :
                    y.append(cor1_list[m])
                #print("y",y)
                y1 = min(y)
                y2 = max(y)

            box_org = [x1,x2,y1,y2]
            #print("box_org",box_org)

            cor_list.append(box_org)
            #print("cor_list",cor_list)

            # roi = image[y: y + h, x: x + w]
            mask = cor_list[0]
            #print("mask", mask)
            x1 = mask[0]
            x2 = mask[1]
            y1 = mask[2]
            y2 = mask[3]
            w = mask[3] - mask[2]
            h = mask[1] - mask[0]


            # merge_img
            if h > w:
                #print("h",h)
                #print("w", w)
                ori_img = yolo_source + "/" + name + ".png"
                ori_img = cv2.imread(ori_img)
                ori_img = cv2.resize(ori_img, dsize=(380, 380), interpolation=cv2.INTER_AREA)
                #cv2.imshow('ori_img', ori_img)
                #cv2.imshow('yolo_img', image)

                back_resize = cv2.resize(back, dsize=(w, h), interpolation=cv2.INTER_AREA)
                ori_imgcopy = ori_img.copy()
                #print(back_resize.shape)
                image[x1:x2, y1:y2] = back_resize
                crop = cv2.subtract(ori_imgcopy, image)
                #cv2.imshow('crop', crop)
                #cv2.waitKey(0)
                cv2.imwrite(yolo_source + "/" + name + "-crop.png", crop)
                #print(yolo_source + "/" + name + "-crop.png")

            elif h < w:
                ori_img = yolo_source + "/" + name + ".png"
                ori_img = cv2.imread(ori_img)
                ori_img = cv2.resize(ori_img, dsize=(380, 380), interpolation=cv2.INTER_AREA)
                #cv2.imshow('ori_img', ori_img)
                #cv2.imshow('yolo_img', image)

                back_resize = cv2.resize(back, dsize=(w, h), interpolation=cv2.INTER_AREA)
                #print(back_resize.shape)
                ori_imgcopy = ori_img.copy()
                image[x1:x2, y1:y2] = back_resize
                crop = cv2.subtract(ori_imgcopy, image)
                #cv2.imshow('crop', crop)
                #cv2.waitKey(0)
                cv2.imwrite(yolo_source + "/" + name + "-crop.png", crop)
                #print(yolo_source + "/" + name + "-crop.png")

    return back_resize




#crop_merge_one("D:/NIH-git/poc1/Yolo_EfficientNet_Inference/dataset/00000002_000-yolo.png")