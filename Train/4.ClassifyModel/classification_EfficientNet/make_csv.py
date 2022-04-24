#save csv("train_covid_croped", "test_covid_croped")
import os
import pandas as pd
import csv

dir = 'D:\\LYL\\final_colorectal_gastric_train\\dataset\\'
train_img_dir = dir+'classification\\train\\'
test_img_dir = dir+'classification\\test\\'
label_dir = dir+'labels\\'

path = []
label = []

for root, dirs, files in os.walk(test_img_dir):
    for file in files:
        if file.endswith(".png"):
            img_path = os.path.join(root, file)
            # print("img_path", img_path)
            if "(" in img_path:
                label_name = img_path.split('\\')[-1][:-7] + '.txt'
            else:
                label_name = img_path.split('\\')[-1][:-4] + '.txt'
            label_path = label_dir+label_name
            # print("label_name", label_name)
            # print("label_path", label_path)

            lines = open(label_path).readlines()
            cls = lines[0][0]
            # print("lines", lines)
            # print("cls", cls)
            # print("\n")

            path.append(img_path)
            label.append(cls)


print("path", path)
print("label", label)

make_csv = pd.DataFrame({"path":path,
                        "label":label})

make_csv.to_csv("test_covid_croped.csv", mode='w')
print(make_csv)

# f.write("path,label \n")
#
# for i in range(len(path)):
#     f.write(path[i] + ',' + label[i] + '\n')
#
# f.close()

