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

from bounding_box_crop import crop_merge_one,crop_merge_all



def Yolo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5-V5.0_best_weight_batch16_epoch20.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='KAGGLE-IMAGE/archive/TEST_YOLO', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='result', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()


    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    save_dir = opt.source

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:

        name = path.split(".")[0]
        #print(name)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            #print(p.name)
            #(save_dir / p.name).mkdir(parents=True, exist_ok=True)
            #save_path = str(save_dir / p.name / p.name)  # img.jpg
            save_path = save_dir
            #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            txt_path = str(save_dir) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            #cv2.imwrite(save_path, img) #원본
            if save_img:
                if dataset.mode == 'image':
                    #cv2.imwrite(save_path, im0)
                    #print(name + "_yolo.png")
                    cv2.imwrite(name + "-yolo.png", im0)

                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')







def EfficientNet():
    parser = argparse.ArgumentParser(description='PyTorch Classification')
    parser.add_argument('--model_name', type=str, default="efficientnet_b4", metavar='S', help='model name')
    parser.add_argument('--model_path', type=str, default="efficientnet_b4.pt", metavar='S', help='model path')
    parser.add_argument('--num_classes', type=int, default=4, metavar='N', help='num classes')
    parser.add_argument('--dataset_dir', type=str, default="D:/NIH-yolo-efficientnet/KAGGLE-IMAGE/archive/TEST_EFFICIENT", metavar='S',
                        help='model path')


    opt = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = Classifier(opt)
    model.load_state_dict(torch.load(opt.model_path))
    model.to(device)
    model.eval()

    source_path = opt.dataset_dir
    #print(source_path)
    source_path_list = os.listdir(source_path)
    #print(source_path_list)
    data_folder_image = []
    image_list = []
    for i in range (len(source_path_list)):

        data_image = source_path + "/" + source_path_list[i] #폴더
        print(data_image)
        #if data_image.endswith("crop.png"):
        #        image_list.append(data_image)
        image_list.append(data_image)

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


    col_name = ["img_name", "prediction", "Negative for Pneumonia", "Typical Appearance",
                "Indeterminate Appearance", "Atypical Appearance"]
    csv_df = pd.DataFrame(csv_list, columns=col_name)
    csv_df.to_csv(source_path + "/efficientnet_classification.csv", index=None)  # 수정; classify 결과 csv파일 이름




def Yolo_Efficient():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5-V5.0_best_weight_batch16_epoch20.pt',help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='KAGGLE-IMAGE/archive/TEST_FULL',help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='KAGGLE-IMAGE/archive/TEST_FULL', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')

    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--model_name', type=str, default="efficientnet_b4", metavar='S', help='model name')
    parser.add_argument('--model_path', type=str, default="efficientnet_b4.pt", metavar='S', help='model path')
    parser.add_argument('--num_classes', type=int, default=4, metavar='N', help='num classes')
    #parser.add_argument('--dataset_dir', type=str, default="KAGGLE-IMAGE/archive/TEST_FULL", metavar='S',help='model path')
    opt = parser.parse_args()

    # Yolo
    print("=====================================Yolo==========================================")
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    save_dir = opt.project

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:

        name = path.split(".")[0]
        # print(name)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # print(p.name)
            # (save_dir / p.name).mkdir(parents=True, exist_ok=True)
            # save_path = str(save_dir / p.name / p.name)  # img.jpg
            save_path = save_dir
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            txt_path = str(save_dir) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # cv2.imwrite(save_path, img) #원본
            if save_img:
                if dataset.mode == 'image':
                    # cv2.imwrite(save_path, im0)
                    # print(name + "_yolo.png")
                    cv2.imwrite(name + "-yolo.png", im0)

                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    print()


    # bounding box crop and resize
    print("=====================================bounding box crop and resize==========================================")
    crop_merge_all(save_path)


    # EfficientNet
    print("=====================================EfficientNet==========================================")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Classifier(opt)
    model.load_state_dict(torch.load(opt.model_path))
    model.to(device)
    model.eval()

    source_path = save_path
    #print(source_path)
    source_path_list = os.listdir(source_path)
    #print(source_path_list)
    data_folder_image = []
    image_list = []
    for i in range(len(source_path_list)):

        data_image = source_path + "/" + source_path_list[i]  # 폴더
        #print(data_image)
        if data_image.endswith("crop.png"):
            image_list.append(data_image)

    #print(image_list)
    dur_sum = 0
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
        if idx == 0:  # 최초 frame은 느리므로 평균에서 제외
            avg_fps = 0
        else:
            dur_sum += dur
            avg_fps = idx / dur_sum

        msg = "avg_fps=%.1f(%.1f %.1f msec) %s class=%d [%.6f %.6f %.6f %.6f]" % (
        avg_fps, fps, dur * 1000, img_path0, class_id, prob[0], prob[1], prob[2], prob[3])
        print(msg)
        image1 = cv2.resize(image0, dsize=(0, 0), fx=0.42, fy=0.42)
        cv2.putText(image1, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("pytorch", image1)

        #print("class_id", class_id)
        #print("class_id type", type(class_id))
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

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            cv2.waitKey(0)


    col_name = ["img_name", "prediction", "Negative for Pneumonia", "Typical Appearance",
                "Indeterminate Appearance", "Atypical Appearance"]
    csv_df = pd.DataFrame(csv_list, columns=col_name)
    csv_df.to_csv(source_path + "/yolo_efficientnet_classification.csv", index=None)  # 수정; classify 결과 csv파일 이름


if __name__ == '__main__':
    #check_requirements(exclude=('pycocotools', 'thop'))
    #with torch.no_grad():
    #    Yolo()

    #EfficientNet()

    Yolo_Efficient()