<div align="center">
  <p>
    <a align="center" href="https://ultralytics.com/yolov5" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png"></a>
  </p>
[英文](README.md)|[简体中文](README.zh-CN.md)<br>
  <br>

# 简介

**方便使用者在Jetson nano上迅速部署YOLOv5模型实现检测任务的开源项目**

# 教程

### 1、Jetson nano搭建YOLOv5环境

### 2、训练自己的数据集

有关训练模型的教程见[训练自己的数据集](https://docs.ultralytics.com)。

### 3、Easy-Yolov5

#### 3.1、jetson nano下克隆本项目

#### 3.2、将步骤2训练好的模型文件 ```.pt```和数据文件```.yaml```复制分别传输到 ```weights```文件夹、```data```文件夹下

#### 3.3、根据项目根目录下的```my_detect.py```脚本中提示内容修改

#### 3.4、运行脚本

```python
python3 my_detect.py
```

