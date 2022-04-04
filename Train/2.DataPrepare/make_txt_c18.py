"""
yolo 모델을 학습시키기위해 필요한 annotation txt 파일을 생성하는 script다.
"""

import json
import os, glob
import time
from tqdm import tqdm
import pydicom
from pydicom import dcmread
import re
import mritopng
from matplotlib import pyplot as plt

def preprocessing_json(json_dir):
    json_list = get_json_list(json_dir)

    for _,json_name in enumerate(tqdm(json_list)):
        data = open_json(json_name)
        dicom_name = json_name[:-11] + '.dcm'
        img_x, img_y = get_dicom_size(dicom_name)
        bbox = get_bbox(data)
        point = get_yolo(bbox, img_x, img_y)
        anot_list = make_anot_list(data, point)
        #folder_name = json_name[:-4]+'txt'
        txt_name = json_name.split('\\')[-1][:-11] + '.txt'
        if anot_list:
            save_txt(txt_save_path, txt_name, anot_list)
        time.sleep(0.005)


def get_dicom_size(dicom_name):
    ds = dcmread(dicom_name)
    #ds.PhotometricInterpretation = 'YBR_FULL'
    img_x, img_y = ds.pixel_array.shape

    #print("dicom size:", img_x, img_y)
    return img_x, img_y


#yolo 모델의 경우 annotation의 형식으로 center(x,y) 좌표와 width, height의 상댓값을 필요로한다.
def get_yolo(bbox_list, img_x, img_y):
    point_list = list()
    x_list = list()
    if len(bbox_list) >= 1:
        for i in range(len(bbox_list)):
            center_x = float(bbox_list[i][2]) / img_x
            center_y = float(bbox_list[i][4]) / img_y
            width = float(bbox_list[i][6]) / img_x
            height = float(bbox_list[i][8]) / img_y

            point = str(center_x) + " " + str(center_y) + " " + str(width) + " " + str(height)
            #print("point", point)

            point_list.append(point)
    else:
        point_list = []

    #print("point_list:", point_list)
    return point_list



def get_stage(json_data):
    json_get = json_data.get('images')
    json_get_anot = json_get[0].get('class')
    return str(json_get_anot)

def open_json(data_path):
    with open(data_path, 'r', encoding='UTF8') as f:
        json_data = json.load(f, strict=False)
    f.close()

    #print("json_data:", json_data)
    return json_data


def get_json_list(json_dir):
    json_list = list()

    print("\n")
    print("Searching for json files...")

    for root, dirs, files in os.walk(json_dir):
        for file in files:
            if file.endswith("_merge.json"):
                json_path = os.path.join(root, file)
                json_list.append(json_path)

    #print(json_list)
    return json_list

def get_bbox(json_data):
    bbox_list = []
    json_get = json_data.get('images')
    bbox = json_get[0].get('boxes')
    if type(bbox) is str:
        print(type(bbox))
        #bbox정보가 string이라서 하는 작업
        b = bbox[1:-1]
        bb = re.split('{|}|: |, ', b)
        n = 10
        bbox_list = [bb[i:i + n] for i in range(0, len(bb), n)]

    #print("bbox_list:", bbox_list)
    #print("len(bbox_list):", len(bbox_list))
    return bbox_list


def make_anot_list(json_data, point_list):
    anot_list = list()
    if len(point_list) == 0:
        point = get_stage(json_data) + " " + str('none')
        anot_list.append(point)
    else:
        for i in range(len(point_list)):
            point = get_stage(json_data) + " " + point_list[i]
            #print("point:", point)
            anot_list.append(point)

    return anot_list

def save_txt(save_path, filename, anot_list):
    print("Start creating text files...")
    #with open(folder_name, 'w') as f:
    with open(save_path + "\\" + filename, 'w') as f:
        f.write('\n'.join(anot_list))
    f.close()


def dicom2png(json_dir, png_save_path):
    count = 0
    for root, dirs, files in os.walk(json_dir):
        for file in files:
            if file.endswith(".dcm"):
                dcm_path = os.path.join(root, file)
                json_path = dcm_path[:-4] + '_merge.json'
                if os.path.isfile(json_path):
                    #png_name = dcm_path.split('\\')[-1][:-4] + '.png'
                    # png_path = png_save_path + "\\" + png_name
                    # d = dcmread(dcm_path)
                    # d.PhotometricInterpretation = 'YBR_FULL'
                    # plt.imsave(png_path, d.pixel_array, cmap=plt.cm.bone)

                    png_name = dcm_path.split('\\')[-1][:-4] + '.png'
                    png_path = png_save_path + png_name
                    mritopng.convert_file(dcm_path, png_path)
                    count += 1
                    print("success", count, png_path)



if __name__ == '__main__':
    txt_save_path = r"../../dataset/labels/"  #(수정)path to save output txt file
    png_save_path = r"../../dataset/images/" #(수정)path to save converted image file(dicom to png)
    json_dir = r"../../dataset/train/"  #(수정)merge json이 존재하는 상위폴더

    #test path
    # json_dir = r"../../dataset/00testtest/"
    # txt_save_path = r"../../dataset/00testtest/labels/"
    # png_save_path = r"../../dataset/00testtest/images/"


    preprocessing_json(json_dir)
    print("끝111111111\n")
    dicom2png(json_dir, png_save_path)
    print("끝22222222222222\n")