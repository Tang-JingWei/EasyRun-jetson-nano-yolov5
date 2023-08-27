# -*- coding: GBK -*-
import threading
import struct
import time
import serial

import argparse
import os
import platform
import sys
from pathlib import Path

import torch


# 导入串口包
sys.path.append("/home/uav/uav-projects/")
import USB1
# 使用GPIO反转LED外设
import numpy as np
from gpio import my_gpio
led_time_cnt = 0
led_toggle = True


FILE = Path(__file__).resolve() 
ROOT = FILE.parents[0]   # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH 
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


# YOLOv5 data pack
def yolo_dataPack(datas):
    data = struct.pack("<bbhb", 
                            0x13,
                            0x15,
                            int(datas),
                            0x52
                            )
    return data


'''
使用说明
    进行特定检测任务，需要注意的参数：
    1. weights: 将权重文件复制进weights文件夹，参数填入自己的权重文件路径
    2. data: 填入dataset的yaml文件，此文件指定了各类检测对象
    3. max_det: 最大检测对象数
    4. classes: 可以过滤筛选检测对象，None表示无筛选
    ...
'''
@smart_inference_mode()
def yolov5_detect(
        weights=ROOT /  'weights/my_2019.pt',  # model.pt path(s)
        data=ROOT / 'data/my_2019.yaml',  # dataset.yaml path
        conf_thres=0.20,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half) # ???????
    # bs = 1
    stride, names, pt = model.stride, model.names, model.pt
    # imgsz = check_img_size(imgsz, s=stride)  # check image size
    # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    # Get the VideoCapture Object
    capture = cv2.VideoCapture(0)

    dt = (Profile(), Profile(), Profile())

    while True:
        ret, frame = capture.read()

        # 1. 按比例缩小图片
        #height, width = frame.shape[:2]
        #newsize = check_img_size(list([width/1, height/1]))
        #frame = cv2.resize(frame, newsize)

        # 2. 指定大小缩小图片
        newsize = [320, 256]  # 需要被32整除
        frame = cv2.resize(frame, newsize)

        height, width, channels = frame.shape # 获取 宽高

        img = torch.from_numpy(frame).to(device)
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        img = img.transpose(2, 3) # 交换通道数和批处理位置
        img = img.transpose(1, 2) 

        # 推理
        with dt[1]:
            pred = model(img, augment=augment, visualize=visualize)

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            s = ''
            annotator = Annotator(frame, line_width=line_thickness, example=str(names))
            if len(det):
                objs = []
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()

                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += str(n.item()) + ' ' + str(names[int(c)]) + ' '  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):  # ???????§????????
                    # print("[det] ",det.detach().cpu().numpy())
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    detect_cx = int(xywh[0]*width)
                    detect_cy = int(xywh[1]*height)
                    detect_width = int(xywh[2]*width)
                    detect_height = int(xywh[3]*width)
                    print("[Objetcts] ","class:",c,names[c], "conf:",int(conf*100),"%", " cx:", detect_cx, " cy:", detect_cy)
                    sendInfo = c+1  # 发送数据为：类型c+1
                    # objs.append([c,int(conf*100),detect_cx,detect_cy])
                    # COMM.write([c,int(conf*100),detect_cx,detect_cy])  # To Do: 百分比传输
            else:
                sendInfo = 0  #未检测到 发送0
            LOGGER.info(f"{'[Inference Time] '}{dt[1].dt * 1E3:.1f}ms")  # 推理时间打印
            print()

        # 串口发送数据
        send_data = yolo_dataPack(sendInfo)
        USB1.COMM.write(send_data)
        time.sleep(0.001)

        # LED反转提示正在发送数据，线程正常执行
        global led_toggle, led_time_cnt
        led_time_cnt = led_time_cnt + 1
        if led_time_cnt >= 5:
            led_time_cnt = 0
            led_toggle = np.invert(led_toggle)
            my_gpio.set_gpio_out(16, led_toggle)

        # 显示图片
        # cv2.imshow('yolo detect by TangJW', frame)
        # if cv2.waitKey(1) == ord('q'):
        #     break

        # 返回
        # return len(det), det


def thread_yolo():
    try:
        print("[Thread Run] ==> YOLO Detect")
        yolov5_detect()
    except Exception as e:
        print(e)
        thread_yolo.join()

if __name__ == "__main__":
    thread_send = threading.Thread(target=thread_yolo)
    thread_send.start()

