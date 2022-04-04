## Readme

# 설치 환경 
CUDA 11.1.1
cuDNN 8.0.4.30
python 3.7.0

#install 'requirements.txt'

#dataset download
- (kaggle: SIIM COVID-19) https://www.kaggle.com/c/siim-covid19-detection
- kaggle NIH Chest X-rays 112,120장 
- CC0: Public Domain - 상업적 이용 가능
- https://www.kaggle.com/nih-chest-xrays/data

# json 내 필요 key 값

<meta2json>
(0008, 0060) Modality                            CS: 'DX'  #modality

(0010, 0010) Patient's Name                      PN: '3a0c965d2601'
(0010, 0020) Patient ID                          LO: '2c00dc1ead80'  #patient ID
(0010, 0040) Patient's Sex                       CS: 'M'  #patient`s sex

(0020, 000d) Study Instance UID                  UI: 000c9c05fd14  #study name
(0020, 000e) Series Instance UID                 UI: e555410bd2cd  #series name

(0008, 0018) SOP Instance UID                    UI: 51759b5579bc  #dicom image name


<csv2json>
boxes
label
Negative for Pneumonia
Typical Appearance
Indeterminate Appearance
Atypical Appearance

# train code 설명
***YOLOv5-v5.0 (poc1\Yolo_EfficientNet_\yolov5-V5.0)
- data\coco128.yaml 수정; train이미지경로/val이미지경로/nc(class개수)/names(class이름) data에 맞게 수정
- train.py; parser부분 '--weights'(pre-trained model), '--data'(data yaml) 경로 수정

*** EfficientNetB4 (poc1\Yolo_EfficientNet_\Train\4.ClassifyModel\classification)
- make_csv.py; 이미지,레이블 경로 부분만 수정 (train에 사용할 csv 생성)
- training.py; parser부분 '--train_path'(train csv), 'test_path'(test csv), '--dataset_dir'(image) 경로 수정


### Train_Model Process
***<2.DataPrepare> - Data Preprocessing
- (1) make_json_real.py (dicom metadata와 data제공 csv를 각각 json으로 변환하여 하나의 json으로 병합하여 저장)
- (2) make_txt_c18.py (json to txt convert for detection-YOLOv5)
- (3) crop_make_folder.py (ROI crop & foldering image by class for classification-EfficientNetB4)
#각각 경로변경 필요; 코드내에 주석 '(수정)'으로 표시해 둔 곳 수정

***<3.DetectModel> - YOLOv5 Model Training
- (4) coco128.yaml (train/val image data path, number of classes, class names setting)
- (5) train.py(Detection-YOLOv5 training-->save model weight file("best.pt"/"last.pt"))

***<4.ClassifyModel> - EfficientNetB4 Model Training
- (6) make_csv.py (create dataset csv -->save csv("train_covid_croped", "test_covid_croped"))
- (7) training.py (Classification-EfficientNetB4 training-->save model weights folder("output_b4_x"))


# inference 코드 설명
- Yolo_Efficient_inference_###.py: yolo, efficientnet, yolo+crop+resize+efficientnet 코드 + 최종 csv 파일 작성 코드
- Yolo_Efficient_inference_one.py: 이미지 하나
- Yolo_Efficient_inference_image.py: 이미지 하나가 yolo-crop-resize-efficientnet
- Yolo_Efficient_inference_function.py: 이미지 전체가 yolo-crop-resize-efficientnet


# 기타
- models, utils: yolo inference에 필요한 models, utils
- black.png: efficientnet inference에 적합한 이미지 크기로 수정할 때 사용하는 검정 배경
- boundingbox 폴더: 샘플용 원본, yolo inference 이미지
- bounding_box_crop.py: 샘플용 바운딩 박스 크롭 코드
- check_shape: 크롭된 바운딩 박스들 이미지 속성 확인 코드
- efficient_model.py: efficientnet model
- efficientnet_b4.pt:  efficientnet weights
- except_data_check.py: yolo-efficientnet 결과물 csv에서 빈칸이 있는 이미지 데이터 확인 
- yolov5-V5.0_best_weight_batch16_epoch20.pt: yolo weights



