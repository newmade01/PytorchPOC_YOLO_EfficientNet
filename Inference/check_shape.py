import os
import cv2

yolo_source = "./KAGGLE-IMAGE/archive/TEST_FULL"
yolo_source_list = os.listdir(yolo_source)
crop_image_list = []
for i in range(len(yolo_source_list)):
    data_image = yolo_source + "/" + yolo_source_list[i]  # 폴더
    print("data: ",data_image)
    if data_image.endswith("crop.png"):
        crop_image_list.append(data_image)
print(crop_image_list)


for i in range(len(crop_image_list)):
    img = crop_image_list[i]
    print(img)
    image = cv2.imread(img)
    print(image.shape)
    print()