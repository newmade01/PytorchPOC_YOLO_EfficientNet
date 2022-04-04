import os
import shutil

data_path = "D:/NIH-yolo-efficientnet/KAGGLE-IMAGE/archive/TOTAL-USE/"
data_list = os.listdir(data_path)

#dst = "D:/NIH-yolo-efficientnet/KAGGLE-IMAGE/archive/TEST_FULL/"
dst = "D:/NIH-yolo-efficientnet/KAGGLE-IMAGE/archive/TEST_YOLO/"
#dst = "D:/NIH-yolo-efficientnet/KAGGLE-IMAGE/archive/TEST_EFFICIENT/"

for i in range(1000):
        img_path = data_path + data_list[i]
        print(img_path)
        print(dst+data_list[i])
        print()
        shutil.copyfile(img_path,dst+data_list[i] )