# YOLOv5

我们采用了YOLOv5作为检测模型。该文件夹内主要提供模型训练方法，主要注意两点：

* data/hyp.scratch.yaml文件内含有各个超参数的设置，可以自行调适
* data/test.yaml用于调整训练数据的目录，默认参照PreProcessing部分处理的结果

```bash
python train.py --img 800 --batch 16 --epochs 200 --weights yolov5x.pt
```

以上是我们训练所用的命令：

* --img设置输入图像大小，理论上稍大一些效果会好
* --batch设置batchsize，越大模型收敛的速度越快效果越好，设置为16时占用显存22G，充分利用3090显卡的性能
* --epoch设置训练的总epoch数目，该数据集在200 epoch后基本收敛完毕
* --weight设置基础模型，可以自行选择，此处选择效果最好的yolov5x

另外还可以通过命令行设置label smooth参数，如下所示：

```bash
python train.py --img 800 --batch 16 --epochs 200 --weights yolov5x.pt --label-smoothing 0.1
```

同样地，可通过手动调整hyp.scratch.yaml内的参数设置mixup=0.2。实验中利用三折交叉验证对上述三种方式进行测试，得到结果如下：

|  测试准确率  | 第一折 | 第二折 | 第三折 | 平均准确率 |
| :----------: | :----: | :----: | :----: | :--------: |
|   原始模型   | 0.843  | 0.838  | 0.867  |   0.849    |
| label smooth | 0.843  | 0.889  | 0.849  |   0.860    |
|    mix up    | 0.870  | 0.900  | 0.865  |   0.878    |

最终采用mixup设置进行模型训练，最终得到模型在测试集上的准确率为0.883。