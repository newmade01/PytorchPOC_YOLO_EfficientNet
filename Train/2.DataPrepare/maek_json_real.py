import pydicom
import matplotlib.pyplot as plt
import pydicom.data
import pandas as pd
import json
from pydicom.dataset import Dataset
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def dicom_list(raw_path):
    dcm_list = list()
    for root, dirs, files in os.walk(raw_path):
        for file in files:
            if file.endswith(".dcm"):
                dcm_path = os.path.join(root, file)
                dcm_list.append(dcm_path)
    #print("dcm_list: ", dcm_list, "\n")

    return dcm_list


def meat2json(dcm):
    print("====start converting meta data to json====")
    print("\n")
    meta2json_list = list()
    ds = pydicom.dcmread(dcm)  # dicom파일 불러오기
    #print("type(ds):", type(ds))  # metaData 출력
    ds_json = ds.to_json_dict()
    meta2json = dcm[:-4] + '_meta.json'
    with open(meta2json, "w") as f:  # json파일 저장위치
        json.dump(ds_json, f)
    #print("meta2json_list: ", meta2json_list, "\n")

    return meta2json


def csv2json(csv_path, dcm_list):
    print("====start converting csv data to json====")
    print("\n")
    train_df = pd.read_csv(csv_path + 'merged_csv.csv')
    csv2json_list = list()
    for i in range(train_df.shape[0]):
        subj_series = train_df.loc[i]  # (subj_series)type; pandas.core.series.Series
        # print("======subject", i, "======")
        # print("subj_series.index: ", subj_series.index)  #['StudyInstanceUID', 'imageID', 'boxes', 'label', 'StudyInstanceUID.1','studyID', 'Negative for Pneumonia', 'Typical Appearance','Indeterminate Appearance', 'Atypical Appearance']
        # print("subj_series.values: ", subj_series.values, "\n")
        # print(subj_series)
        # print("==========================\n")

        subj_dict = subj_series.to_dict()  # type; dict

        # subj_json = json.dumps(subj_dict)  #convert dictionary to json

        cls_yn = subj_dict.get('Negative for Pneumonia')

        if not pd.isna(cls_yn):
            for k in range(len(dcm_list)):
                if subj_dict.get('imageID')[:-6] in dcm_list[k]:
                    path_split = dcm_list[k].split('\\')
                    # print("path_split: ", path_split, "\n")
                    save_path = path_split[:-1]
                    # print("save_path: ", save_path, "\n")
                    path_join = '/'.join(save_path)
                    # print("path_join: ", path_join, "\n")
                    csv2json_dir = path_join + '/' + subj_dict.get('imageID')[:-6] + '_csv.json'
                    # print("csv2json_dir: ", csv2json_dir, "\n")

                    with open(csv2json_dir, "w") as f:  # json파일 저장위치
                        json.dump(subj_dict, f)

                    csv2json_list.append(csv2json_dir)

    #print("csv2json_list: ", csv2json_list, "\n")
    return csv2json_list


def get_meta2json(meta2json):
    print("====start getting meta2json info====")
    with open(meta2json, 'r', encoding='UTF8') as f:
        meta_json = json.load(f, strict=False)  # type; dictionary
            # print(type(meta_json))
    Modality = meta_json.get("00080060")  # type; dictionary
    Modality = Modality.get("Value")
    Modality = Modality[0]
    Patient_ID = meta_json.get("00100020")
    Patient_ID = Patient_ID.get("Value")
    Patient_ID = Patient_ID[0]
    Sex = meta_json.get("00100040")
    Sex = Sex.get("Value")
    Sex = Sex[0]
    Study_Name = meta_json.get("0020000D")
    Study_Name = Study_Name.get("Value")
    Study_Name = Study_Name[0]
    Series = meta_json.get("0020000E")
    Series = Series.get("Value")
    Series = Series[0]
    Dicom = meta_json.get("00080018")
    Dicom = Dicom.get("Value")
    Dicom = Dicom[0]

    metajson = {'Modality': Modality, 'Patient_ID': Patient_ID, 'Sex': Sex, 'Study_Name': Study_Name,
                'Series': Series, 'Dicom': Dicom}
    # print("metajson: ",metajson)

    print("metajson", metajson)
    print("\n")
    return metajson


def get_csv2json(csv2json_list):
    print("====start getting csv2json info====")
    csvjson_list = list()
    for i in range(len(csv2json_list)):
        with open(csv2json_list[i], 'r', encoding='UTF8') as f:
            csv_json = json.load(f, strict=False)  # type; dictionary

        Dicom_csv = csv_json.get("imageID")
        bbox = csv_json.get("boxes")
        label = csv_json.get("label")
        cls0 = csv_json.get("Negative for Pneumonia")
        if type(cls0) is str:
            cls0 = int(cls0)
        cls1 = csv_json.get("Typical Appearance")
        if type(cls1) is str:
            cls1 = int(cls1)
        cls2 = csv_json.get("Indeterminate Appearance")
        if type(cls2) is str:
            cls2 = int(cls2)
        cls3 = csv_json.get("Atypical Appearance")
        if type(cls3) is str:
            cls3 = int(cls3)

        csvjson = {'Dicom': Dicom_csv[:-6], 'boxes': bbox, 'label': label, 'Negative_for_Pneumonia': cls0,
                   'Typical_Appearance': cls1, 'Indeterminate_Appearance': cls2, 'Atypical_Appearance': cls3}
        # print("csvjson: ", csvjson)
        csvjson_list.append(csvjson)

    print(csvjson_list)
    print("\n")
    return csvjson_list

def write_cls(csv2json_list):
    print("=====start writhing class====")
    cls_list = list()
    for i in csv2json_list:
        cls0 = i.get('Negative_for_Pneumonia')
        cls0 = int(cls0)
        cls1 = i.get('Typical_Appearance')
        cls1 = int(cls1)
        cls2 = i.get('Indeterminate_Appearance')
        cls2 = int(cls2)
        cls3 = i.get('Atypical_Appearance')
        cls3 = int(cls3)

        if cls0 == 1:
            cls0 = 0
            cls = {'class' : cls0}
        elif cls1 == 1:
            cls1 = 1
            cls = {'class' : cls1}
        elif cls2 == 1:
            cls2 = 2
            cls = {'class' : cls2}
        else:
            cls3 = 3
            cls = {'class' : cls3}

        cls_list.append(cls)

    print("cls_list: ",cls_list)
    print("\n")
    return cls_list

def merge_json(csv2json_list, metajson, csvjson_list, cls_list):
    print("====start merging jsons====")
    for i in range(len(csvjson_list)):
        if metajson.get("Dicom") == csvjson_list[i].get("Dicom"):
            aa = metajson
            bb = csvjson_list[i]
            cc = cls_list[i]
            # match = csvjson_list[i].get("Dicom_csv")
            # print(metajson_list[i].get("Dicom"))
            # print(csvjson_list[i].get("Dicom"))

            dd = dict(bb, **cc)
            merge = dict(aa, **dd)
            merge = {"images": [merge]}
            # img = json.get('images')

            dir = csv2json_list[i][:-8] + 'merge.json'
            #print("dir:", dir)
            #print("\n")
            with open(dir, "w") as f:  # json파일 저장위치
                json.dump(merge, f)
            print("final_json:", merge)
            print(type(merge))
            print("\n")


def make_json(raw_path, csv_path):
    dcm_list = dicom_list(raw_path)
    csv2json_list = csv2json(csv_path, dcm_list)
    csvjson_list = get_csv2json(csv2json_list)
    cls_list = write_cls(csvjson_list)

    for _, dcm in enumerate(dcm_list):
        meta2json = meat2json(dcm)
        metajson = get_meta2json(meta2json)
        merge_json(csv2json_list, metajson, csvjson_list, cls_list)


if __name__ == '__main__':
    csv_path = "..\\..\\dataset\\"
    raw_path = csv_path+'train\\'
    #raw_path = csv_path+'00testtest\\'

    make_json(raw_path, csv_path)
    print("================Process End======================")