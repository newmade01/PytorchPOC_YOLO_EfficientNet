import shutil
import pandas as pd
import numpy as np
import cv2
from bounding_box_crop import crop_merge_one
import torch
import argparse
import albumentations as A
import torch.nn.functional as F
from efficient_model import Classifier
from albumentations.pytorch import ToTensorV2
import time
import os

def EfficientNet_check(img):
    parser = argparse.ArgumentParser(description='PyTorch Classification')
    parser.add_argument('--model_name', type=str, default="efficientnet_b4", metavar='S', help='model name')
    parser.add_argument('--model_path', type=str, default="./weights/efficientnet_b4_fold1_best_accuracy.pt", metavar='S', help='model path')
    parser.add_argument('--num_classes', type=int, default=4, metavar='N', help='num classes')


    opt = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Classifier(opt)
    model.load_state_dict(torch.load(opt.model_path))
    model.to(device)
    model.eval()

    print("efficientnet img: ", img)
    image_list = []
    image_list.append(img)


    #print(image_list)
    dur_sum=0
    csv_list = []
    for idx, img_path in enumerate(image_list):
        transform = A.Compose([
            A.Resize(height=380, width=380),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        image0 = cv2.imread(img_path)
        img_path0 = os.path.basename(img_path)

        start_time = time.time()

        image = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
        image = transform(image=image)['image']
        image = image.unsqueeze(0)
        input = image.to(device)
        with torch.no_grad():
            pred = model(input)

            prob = F.softmax(pred, dim=1)
            prob = prob.cpu().data.numpy()[0]

        end_time = time.time()
        dur = end_time - start_time
        class_id = prob.argmax()
        fps = 1 / dur
        if idx==0 : # 최초 frame은 느리므로 평균에서 제외
            avg_fps = 0
        else :
            dur_sum += dur
            avg_fps = idx  / dur_sum

        msg = "avg_fps=%.1f(%.1f %.1f msec) %s class=%d [%.6f %.6f %.6f %.6f]"%(avg_fps, fps, dur*1000, img_path0, class_id, prob[0], prob[1], prob[2], prob[3])
        print(msg)
        image1 = cv2.resize(image0,dsize=(0,0), fx=0.42,fy=0.42)
        cv2.putText(image1, msg, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        #print("class_id",class_id)
        #print("class_id type",type(class_id))
        class_name = ""
        if str(class_id) == "0":
            class_name = "Negative for Pneumonia"
        elif str(class_id) == "1":
            class_name = "Typical Appearance"
        elif str(class_id) == "2":
            class_name = "Indeterminate Appearance"
        elif str(class_id) == "3":
            class_name = "Atypical Appearance"
        value = (img_path0, class_name, prob[0], prob[1], prob[2], prob[3])
        csv_list.append(value)

        #print(csv_list)
        cv2.imshow("pytorch", image1)
        key=cv2.waitKey(1)
        if key==27 :
            break
        elif key==32 :
            cv2.waitKey(0)


    return csv_list






csv_recover = pd.read_csv("./yolo_efficientnet_classification.csv") #, encoding="utf-8")
#print(csv_recover.columns)
#print(csv_recover.head())
#print(csv_recover["prediction"].head(40))
csv_recover = csv_recover.fillna('')
lost_image_list=[]
for i in range(len(csv_recover["prediction"])):
    #print(i)
    #print(csv_recover["prediction"][0])
    if csv_recover["prediction"][i] == "":
        #print(csv_recover["img_name"][i])
        lost_image_list.append(csv_recover["img_name"][i])

print("빠진 이미지           :",lost_image_list)


'''
##작성중##
for i in range(len(lost_image_list)):
    lost_image=lost_image_list[i]
    print("빠진 이미지       :",lost_image)
    ori_img = lost_image[:-9]
    print("빠진 이미지의 yolo:",ori_img+ "-yolo.png   ---> crop ")

    #crop
    crop_merge_one(ori_img + "-yolo.png")

    #efficientnet
    data_image = ori_img + "-crop.png"  # 폴더
    print("빠진 이미지의 crop:", ori_img + "-crop.png   ---> crop 완료")

    csv_list = EfficientNet_check(data_image)
    print(csv_list)
'''
