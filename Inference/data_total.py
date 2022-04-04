import os
import shutil

'''
image_folder =[]
data_path = "D:/kaggle-covid-/KAGGLE-IMAGE/archive/"
data_list = os.listdir(data_path)
dst = data_path + "TOTAL/"

for i in range(len(data_list)):
    if data_list[i].startswith("images"):
        folders = data_path + data_list[i] + "/images"
        image_folder.append(folders)

for j in range(len(image_folder)):
    image_list = os.listdir(image_folder[j])
    for k in range(len(image_list)):
        image_full_path = image_folder[j] + "/" + image_list[k]
        print(image_full_path)
        print(dst+image_list[k])
        print()
        shutil.copyfile(image_full_path,dst+image_list[k])
'''







#data_path = "D:/kaggle-covid-/KAGGLE-IMAGE/archive/images_012/images/"
data_path = "D:/kaggle-covid-"
data_list = os.listdir(data_path)
dst = "D:/kaggle-covid-"

for i in range(len(data_list)):
    image_full_path = data_path + data_list[i]
    print(image_full_path)
    print(dst+data_list[i])
    print()
    shutil.copyfile(image_full_path,dst+data_list[i])
