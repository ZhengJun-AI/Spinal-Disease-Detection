## Resnet-50

我们采用了Resnet-50作为分类模型。该文件主要提供模型训练方法：

先根据PreProcessing部分生成增强各类别图像，然后按照如下格式放入images文件夹中，其中test文件夹中图像为不做任何增强处理的原切割图像：

```
└── images
    ├── train
    |	├── c1
    |	|   └── v_1
    |	|   └── v_2
    |	|   └── V1
    |	|   └── V2
    |	|   └── V3
    |	|   └── V4
    |	|   └── V5only
    |	└── c2
    |	    └── noV5
    |	    └── V5
    └── test
    	├── c1
    	|   └── v_1
    	|   └── v_2
    	|   └── V1
    	|   └── V2
    	|   └── V3
    	|   └── V4
    	|   └── V5only
    	└── c2
    	    └── noV5
    	    └── V5

```

直接运行train.py文件即可开始训练，train.py文件内含有各个超参数的设置，可以自行调试，以下为我们训练所用的命令：

```bash
python train.py -n 7 -train ./images/train/c1 -test ./images/test/c1 --epoch 100
```

以下为可通过命令行修改参数：

* -n/--numclass设置分类数目，在本任务中分别为c1中的7分类和c2中的2分类
* -train/--train_data_dir设置训练集数据目录路径
* -test/--test_data_dir设置测试集数据目录路径
* -name/--model_name设置训练模型的名字，方便保存模型文件及训练结果图片，默认为"mymodel"
* -e/--epoch设置训练的总epoch数目，该数据集在100 epoch后基本收敛完毕
* -b/--batch_size设置训练的batch_size，越大模型收敛的速度越快，默认为512，效果最佳，加上resize为64时只占用显存3G
* -l/--learning_rate设置训练学习率，默认为0.001，效果最佳
* -r/--resize设置图片Resize大小，默认为64，效果最佳

运行后自动保存如下图训练结果：

<img src=".\c1_accuracy.png" alt="c1_accuracy_curve-newrot" style="zoom:80%;" />

<img src=".\c1_loss.png" alt="c1_loss_curve-newrot" style="zoom:80%;" />

最终得到模型在测试集上的准确率为0.6539。