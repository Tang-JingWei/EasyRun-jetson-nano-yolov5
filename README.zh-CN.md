<div align="center">
  <p>
    <a align="center" href="https://ultralytics.com/yolov5" target="_blank">
      <img width="100%" src="https://gitee.com/JingWei1234/project-picture/raw/master/naon%20yolo.png"></a>
  </p>
</div>



# 简介

本项目旨在方便使用者在Jetson nano上迅速部署YOLOv5模型实现检测任务的开源项目。YOLOv5原项目可以实现检测到目标，但是不会直接返回检测目标的一些信息，大白话就是对于初学者，很难理解懂他的源码，比如我要知道检测到的目标的信息，是哪个变量？其次，对于很多场景例如比赛我们不需要官方给出的很多参数，许多代码其实是可以优化掉的，去繁就简很重要。因此，为了开发效率，我在理解yolov5源码框架后，优化了很多代码，留下核心部分，并就作为我自己完成目标检测任务的框架。现开源供大家参考一下，可能在不同的机器上会有不同问题，可以提issues，也请各位大佬指正。

------



# 特点

- [ ] *可以自定义按比例、指定大小缩放视频流传来的图片大小，根据实际预期调整*
- [ ] *可以直接返回检测目标的中心XY坐标、长宽信息、类型、置信度、推理时间*
- [ ] *串口发送数据，结合数据打包函数发送以上任意想要发送的数据*
- [ ] *显示实时检测的图片，注释或者取消注释即可（注释掉显示会稍稍减少性能占用）*

------



# 教程

### 1、Jetson nano搭建YOLOv5环境

自行搜索、搭好YOLOv5环境测试



### 2、训练自己的数据集

有关训练YOLOv5模型的教程推荐：

[YOLOv5训练自己的数据集详解](https://blog.csdn.net/weixin_55073640/article/details/122874005)。



### 3、Easy-Yolov5 RUNS!

#### 3.1、Jetson nano下克隆本项目

```
git clone https://gitee.com/JingWei1234/easy-run-jetson-nano-yolov5.git
```

#### 3.2、将步骤2训练好的模型文件 ```.pt```和数据文件```.yaml```复制分别传输到 ```weights```文件夹、```data```文件夹下

#### 3.3、根据项目根目录下的```my_detect.py```脚本中提示内容修改

```python
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
        weights=ROOT /  'weights/my_abc.pt',  # model.pt path(s)
        data=ROOT / 'data/my_abc.yaml',  # dataset.yaml path
        conf_thres=0.5,  # confidence threshold
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
```



#### 3.4、运行脚本

```python
python3 my_detect.py
```

