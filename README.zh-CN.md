<div align="center">
  <p>
    <a align="center" href="https://ultralytics.com/yolov5" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.png"></a>
  </p>
[英文](README.md)|[简体中文](README.zh-CN.md)<br>

  <br>

YOLOv5 🚀 是世界上最受欢迎的视觉 AI，代表<a href="https://ultralytics.com"> Ultralytics </a>对未来视觉 AI 方法的开源研究，结合在数千小时的研究和开发中积累的经验教训和最佳实践；


## <div align="center">文档</div>

有关训练、测试和部署的完整文档见[YOLOv5 文档](https://docs.ultralytics.com)。请参阅下面的快速入门示例。

<details open>
<summary>安装</summary>

克隆 repo，并要求在 [**Python>=3.7.0**](https://www.python.org/) 环境中安装 [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) ，且要求 [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/) 。

```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

</details>

<details>
<summary>推理</summary>

使用 YOLOv5 [PyTorch Hub](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading) 推理。最新 [模型](https://github.com/ultralytics/yolov5/tree/master/models) 将自动的从
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) 中下载。

```python
import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img = "https://ultralytics.com/images/zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>

<details>
<summary>使用 detect.py 推理</summary>

`detect.py` 在各种来源上运行推理， [模型](https://github.com/ultralytics/yolov5/tree/master/models) 自动从
最新的YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) 中下载，并将结果保存到 `runs/detect` 。

```bash
python detect.py --weights yolov5s.pt --source 0                               # webcam
                                               img.jpg                         # image
                                               vid.mp4                         # video
                                               screen                          # screenshot
                                               path/                           # directory
                                               list.txt                        # list of images
                                               list.streams                    # list of streams
                                               'path/*.jpg'                    # glob
                                               'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                               'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>训练</summary>
下面的命令重现 YOLOv5 在 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) 数据集上的结果。
最新的 [模型](https://github.com/ultralytics/yolov5/tree/master/models) 和 [数据集](https://github.com/ultralytics/yolov5/tree/master/data)
将自动的从 YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) 中下载。
YOLOv5n/s/m/l/x 在 V100 GPU 的训练时间为 1/2/4/6/8 天（ [多GPU](https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training) 训练速度更快）。
尽可能使用更大的 `--batch-size` ，或通过 `--batch-size -1` 实现
YOLOv5 [自动批处理](https://github.com/ultralytics/yolov5/pull/5092) 。下方显示的 batchsize 适用于 V100-16GB。

```bash
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                 yolov5s                    64
                                                                 yolov5m                    40
                                                                 yolov5l                    24
                                                                 yolov5x                    16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>

<details open>
<summary>教程</summary>

- [训练自定义数据](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data) 🚀 推荐
- [获得最佳训练结果的技巧](https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results) ☘️
- [多GPU训练](https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training)
- [PyTorch Hub](https://docs.ultralytics.com/yolov5/tutorials/pytorch_hub_model_loading) 🌟 新
- [TFLite，ONNX，CoreML，TensorRT导出](https://docs.ultralytics.com/yolov5/tutorials/model_export) 🚀
- [NVIDIA Jetson平台部署](https://docs.ultralytics.com/yolov5/tutorials/running_on_jetson_nano) 🌟 新
- [测试时增强 (TTA)](https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation)
- [模型集成](https://docs.ultralytics.com/yolov5/tutorials/model_ensembling)
- [模型剪枝/稀疏](https://docs.ultralytics.com/yolov5/tutorials/model_pruning_and_sparsity)
- [超参数进化](https://docs.ultralytics.com/yolov5/tutorials/hyperparameter_evolution)
- [冻结层的迁移学习](https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers)
- [架构概述](https://docs.ultralytics.com/yolov5/tutorials/architecture_description) 🌟 新
- [Roboflow用于数据集、标注和主动学习](https://docs.ultralytics.com/yolov5/tutorials/roboflow_datasets_integration)
- [ClearML日志记录](https://docs.ultralytics.com/yolov5/tutorials/clearml_logging_integration) 🌟 新
- [使用Neural Magic的Deepsparse的YOLOv5](https://docs.ultralytics.com/yolov5/tutorials/neural_magic_pruning_quantization) 🌟 新
- [Comet日志记录](https://docs.ultralytics.com/yolov5/tutorials/comet_logging_integration) 🌟 新

</details>
