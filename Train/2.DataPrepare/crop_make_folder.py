import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import glob



#raw_path = "../00testtest/" #(수정)image와 label 폴더가 존재하는 상위폴더
raw_path = "D:/LYL/final_colorectal_gastric_train/dataset/"
#raw_path = "../00testtest/zzz/"
dst_folder = raw_path+"classification" #(수정)class별 폴더를 생성할 상위폴더
#dst_folder2 =raw_path+"test/" #(수정)
image_path2 = raw_path+"images/" #(수정)image파일 존재하는 위치
label_path2 = raw_path+"labels/" #(수정)label txt파일이 존재하는 위치

#dst_folder = "D:/AIhub/colorectal/inference"
#dst_folder2 = "D:/AIhub/colorectal/inference/"

num_of_class = 4 #(수정)class개수 --> 이 숫자에 따라 폴더 class별 폴더 생성

def crop_merge_img(img, line):
    global back_resize

    back_dir = 'back.png'
    back = cv2.imread(back_dir)
    img_size = img.shape

    if not line[1] == 'none':
        center_x = float(line[1])
        center_y = float(line[2])
        width = float(line[3])
        height = float(line[4])
        #print(center_x, center_y, width, height)
        #print("\n")

        #crop좌표
        x1 = int((center_x) * img_size[0])
        x2 = int((center_x + width) * img_size[0])
        w = x2-x1
        print("x1", x1, "x2", x2)
        print("w", w)

        y1 = int((center_y) * img_size[1])
        y2 = int((center_y + height) * img_size[1])
        h = y2 - y1
        print("y1", y1, "y2", y2)
        print("h", h)

        # merge_img
        roi = img[y1:y2, x1:x2]
        print("roi.shape", roi.shape)

        if h > w:
            back_resize = cv2.resize(back, dsize=(h, h), interpolation=cv2.INTER_AREA)
            print("back.shape", back_resize.shape)
            # cv2.imshow('background', back_resize)
            # cv2.waitKey(0)

            back_resize[0:h, int(h / 2 - w / 2):int(h / 2 + w / 2)] = roi
            # print("merge.shape", back_resize.shape)
            # print("\n")
            # cv2.imshow('merge_img', back_resize)
            # cv2.waitKey(0)

        else:
            back_resize = cv2.resize(back, dsize=(w, w), interpolation=cv2.INTER_AREA)
            # print("back.shape", back_resize.shape)
            # cv2.imshow('background', back_resize)
            # cv2.waitKey(0)

            back_resize[int(w / 2 - h / 2):int(w / 2 + h / 2), 0:w] = roi
            # print("merge.shape", back_resize.shape)
            # print("\n")
            # cv2.imshow('merge_img', back_resize)
            # cv2.waitKey(0)

    else:
        back_resize = img

    return back_resize


for c in range(0, num_of_class):
  d = ""+str(c)
  path = os.path.join(dst_folder, d)
  #print(path)
  #os.mkdir(path)
  try:
      os.mkdir(path)
  except FileExistsError:
      pass

image_list = os.listdir(image_path2)
#print("image_list ", image_list)
label_list = os.listdir(label_path2)
#print("label_list ", label_list)

for i in range(len(label_list)):
    image_path3 = image_path2 + label_list[i][:-4] + ".png"
    print("image_path3 ", image_path3)
    label_path3 = label_path2 + label_list[i]
    print("label_path3 ", label_path3)

    img = cv2.imread(image_path3)
    lines = open(label_path3).readlines()
    print("lines ", lines)

    uniq = 0
    for line in lines:
        line = line.split()
        # print("line ", line)
        for j in range(0, len(line)):
            # cls = coord[0]
            # print("cls ", cls)

            final = crop_merge_img(img, line)
            img_name = label_list[i][:-4]

        image_dd = dst_folder + "/" + str(int(line[0])) + "/" + img_name + ".png"

        if os.path.isfile(image_dd):
            uniq += 1
            img_name = img_name + '(' + str(uniq) + ')'
            image_dd = dst_folder + "/" + str(int(line[0])) + "/" + img_name + ".png"

        print("image_dd ", image_dd)
        # cv2.imwrite(image_dd, final)
        print("\n")
        # if os.path.isfile(image_dd):
        #    continue



