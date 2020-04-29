# 基于pytorch实现Pointrend的语义分割代码
## 介绍
这个代码主要是实现了将Pointrend上采样模块的“渲染”运用到语义分割任务中。//
Pointrend关键模块的详解请看博客：- [语义分割之PointRend论文与源码解读](https://blog.csdn.net/weixin_42028608/article/details/105379233)

![](docs/images/demo.png)

## 环境
- python 3
- torch >= 1.1.0
- torchvision
- pyyaml
- Pillow
- numpy

## 安装
```
python setup.py develop
```
如果你不想用CCNet，你不需要安装，只需要在```segmentron/models/__init__.py```文件将```from .ccnet import CCNet```注释掉

## 数据集准备
支持 cityscape, coco, voc, ade20k now.

Please refer to [DATA_PREPARE.md](docs/DATA_PREPARE.md) for dataset preparation.

## 预训练的基础模型

预训练的基础模型将自动保存在(```~/.cache/torch/checkpoints/```).

## 代码结构
```
├── configs    # yaml config file
├── segmentron # core code
├── tools      # train eval code
└── datasets   # put datasets here 
```

## 训练
### 使用单GPU训练Pointrend模型
```
CUDA_VISIBLE_DEVICES=0 python -u tools/train.py --config-file configs/cityscapes_pointrend_deeplabv3_plus.yaml
```
### 使用多GPU训练Pointrend模型
```
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

## 评估
### 使用单GPU训练进行评估
你需要将你训练好的模型放于```TEST.TEST_MODEL_PATH```
```
CUDA_VISIBLE_DEVICES=0 python -u ./tools/eval.py --config-file configs/cityscapes_pointrend_deeplabv3_plus.yaml \
TEST.TEST_MODEL_PATH your_test_model_path

```
### 使用多GPU训练进行评估
```
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_test.sh ${CONFIG_FILE} ${GPU_NUM} \
TEST.TEST_MODEL_PATH your_test_model_path
```

## 参考
- [SegmenTron](https://github.com/LikeLy-Journey/SegmenTron)
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [detectron2](https://github.com/facebookresearch/detectron2)
- [gloun-cv](https://github.com/dmlc/gluon-cv)
