import yolov5_usbDT  # 串口传输
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

FILE = Path(__file__).resolve() # 获取当前脚本的绝对路径
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

def yolo_dataPack(datas):
    data = struct.pack("<bbhhhhhhb", 
                            0x13,
                            0x14,
                            int(datas[0].x * 100),
                            int(datas[0].y * 100),
                            int(datas[0].z * 100),
                            int(datas[1].x * 100),
                            int(datas[1].y * 100),
                            int(datas[1].z * 100),
                            0x52
                            )
    return data

# ----------------- 路检 --------------------
yolo_log = open(file="LOG_damageRoad.txt", mode='w+', encoding="utf-8")  # a+ 模式不覆盖  w+ 覆盖
UAV_position_x = 1
UAV_position_y = 1
# -------------------------------------------

'''
使用说明：
    进行特定检测任务，需要注意的参数：
    1、weights。填入自己的模型文件路径。
    2、data。填入dataset的yaml文件，因为此文件指定了各类别检测对象。
    3、max_det。最大检测对象数。
    4、classes。可以过滤筛选检测对象。None表示无筛选
    ...
'''
@smart_inference_mode()
def yolov5_detect(
        weights=ROOT / 'yolov5s_damageRoad.pt',  # model.pt path(s)
        source=0,  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/damage_road.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half) # 载入模型
    # bs = 1
    stride, names, pt = model.stride, model.names, model.pt
    # imgsz = check_img_size(imgsz, s=stride)  # check image size
    # model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

    # Get the VideoCapture Object
    capture = cv2.VideoCapture(0)

    dt = (Profile(), Profile(), Profile())

    # -------------- 路检 ----------------
    yolo_log.write(time.strftime("检测时间：%Y-%m-%d-%H:%M:%S \n", time.localtime()) + '\n')
    yolo_log.write('X坐标' + '\t' + 'Y坐标' + '\t' + '破损类型' + '\t' + '置信度' + '\r\n')
    #------------------------------------

    while True:
        ret, frame = capture.read()

        # 1-按比例缩小图片
        # height, width = frame.shape[:2]
        # newsize = check_img_size(list([width/2, height/2]))
        # frame = cv2.resize(frame, newsize)

        # 2-指定大小缩小图片
        newsize = [320, 256]  # 需要被32整除
        frame = cv2.resize(frame, newsize)

        height, width, channels = frame.shape # 获取图像宽高

        img = torch.from_numpy(frame).to(device)
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        img = img.transpose(2, 3) # 交换通道数和批处理大小的位置
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
                for *xyxy, conf, cls in reversed(det):  # 遍历所有检测到的对象
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
                    #------------------- 路检 --------------------
                    global UAV_position_x, UAV_position_y
                    log = ''
                    log += str(int((UAV_position_x))) + '  '
                    log += str(int((UAV_position_y))) + '  '
                    log += str(names[c]) + '  '
                    log += str(int((int(conf*100)))) + '\n'
                    yolo_log.write(log)
                    yolo_log.flush()
                    #---------------------------------------------
                    # objs.append([c,int(conf*100),detect_cx,detect_cy])
                    # COMM.write([c,int(conf*100),detect_cx,detect_cy])  # To Do: 改成百分比传输

        LOGGER.info(f"{'[Inference Time] '}{dt[1].dt * 1E3:.1f}ms")  # 打印推理时间
        print()

        # 串口发送数据
        # send_data = yolo_dataPack(objs)
        # COMM.write(send_data)
        # time.sleep(0.01)

        # 显示图片
        # cv2.imshow('yolo detect by TangJW', frame)
        # if cv2.waitKey(1) == ord('q'):
        #     break

        # 返回
        # return len(det), det


def thread_yolo():
    try:
        yolov5_detect()
    except Exception as e:
        print(e)
        thread_yolo.join()
        yolo_log.close()
        COMM.close()
        print("[close] " + COMM.name + " close")


COMM = serial.Serial()  # 定义串口对象

if __name__ == "__main__":
    # 开串口
    COMM = serial.Serial('/dev/usb_1', 115200, timeout=0.01)
    if COMM.isOpen():
        print("[open] " + '/dev/usb_1', "open success")
    else:
        print("[open] open failed")
        exit(0)

    thread_send = threading.Thread(target=thread_yolo)
    thread_send.start()

    # yolo_log.write(time.strftime("检测时间：%Y-%m-%d-%H:%M:%S \n", time.localtime()) + '\r\n')
    # yolo_log.write('X坐标' + '\t' + 'Y坐标' + '\t' + '破损类型' + '\t' + '置信度' + '\r\n')
    # while True:
    #     print('hello1')
    #     log = ''
    #     log += str(int((UAV_position_x))) + '\t'
    #     log += str(int((UAV_position_y))) + '\t'
    #     log += str('D40') + '\t'
    #     log += str(int((int(75)))) + '\n'
    #     yolo_log.write(log)
    #     time.sleep(0.1)
